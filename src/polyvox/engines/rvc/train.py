"""Training an RVC v2 voice (40 kHz, with pitch) from a dataset's clips: features per clip
(40 kHz waveform, linear spectrogram, ContentVec features, RMVPE pitch), then the VITS-style GAN
from RVC's base models (f0G40k / f0D40k): mel L1 x 45 + KL + feature matching + adversarial
(least squares), AdamW 1e-4 (0.8, 0.99), learning rate x 0.999875 per epoch, random 32-frame
windows of each clip, bf16 autocast. Saved voices use RVC's file format (fp16 weights,
weight_g / weight_v), so RVC's WebUI and Applio load them too.
"""

import math
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .model import Config, Synthesizer

N_FFT, HOP_40K, N_MELS = 2048, 400, 125
SEGMENT = 32  # frames (12800 samples) per training window
MAX_FRAMES = 900  # RVC keeps at most 9 s of a clip
C_MEL, C_KL = 45.0, 1.0
HIGHPASS_40K = signal.butter(N=5, Wn=48, btype="high", fs=40000)


# ---------------------------------------------------------------- weight norm


def norm_layers(model: Synthesizer) -> list[nn.Module]:
    """The layers RVC trains with weight norm (the rest are plain)."""
    layers = []
    for wn in [model.enc_q.enc] + [f.enc for f in model.flow.flows if hasattr(f, "enc")]:
        layers += [wn.cond_layer, *wn.in_layers, *wn.res_skip_layers]
    layers += list(model.dec.ups)
    for block in model.dec.resblocks:
        layers += [*block.convs1, *block.convs2]
    return layers


def with_weight_norm(model: nn.Module, layers: list[nn.Module]) -> nn.Module:
    for layer in layers:
        weight_norm(layer)  # loads weight_g / weight_v state dicts too (torch's compat hook)
    return model


def plain_state(model: nn.Module) -> dict:
    """State dict in RVC's names: parametrized weights as weight_g / weight_v."""
    out = {}
    for key, value in model.state_dict().items():
        key = key.replace("parametrizations.weight.original0", "weight_g").replace(
            "parametrizations.weight.original1", "weight_v"
        )
        out[key] = value
    return out


# ---------------------------------------------------------------- discriminator


class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            weight_norm(c)
            for c in (
                nn.Conv1d(1, 16, 15, 1, padding=7),
                nn.Conv1d(16, 64, 41, 4, groups=4, padding=20),
                nn.Conv1d(64, 256, 41, 4, groups=16, padding=20),
                nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20),
                nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20),
                nn.Conv1d(1024, 1024, 5, 1, padding=2),
            )
        )
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class DiscriminatorP(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chans = [1, 32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chans[i], chans[i + 1], (5, 1), (strides[i], 1), padding=(2, 0))) for i in range(5)
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), "reflect")
        x = x.view(b, c, -1, self.period)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11, 17, 23, 37)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS()] + [DiscriminatorP(p) for p in periods])

    def forward(self, y, y_hat):
        """Real and generated audio in one pass (each weight computed and each kernel launched once)."""
        n = y.shape[0]
        real, fake, fmap_r, fmap_g = [], [], [], []
        both = torch.cat([y, y_hat])
        for d in self.discriminators:
            out, fmap = d(both)
            real.append(out[:n])
            fake.append(out[n:])
            fmap_r.append([f[:n] for f in fmap])
            fmap_g.append([f[n:] for f in fmap])
        return real, fake, fmap_r, fmap_g


# ---------------------------------------------------------------- spectrograms and losses


class Spectrograms:
    def __init__(self, sample_rate: int, device):
        self.window = torch.hann_window(N_FFT, device=device)
        mel = librosa.filters.mel(sr=sample_rate, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=None)
        self.mel_basis = torch.from_numpy(mel).float().to(device)

    def linear(self, y: torch.Tensor) -> torch.Tensor:  # [b, samples] -> [b, 1025, frames]
        pad = (N_FFT - HOP_40K) // 2
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(y, N_FFT, HOP_40K, N_FFT, window=self.window, center=False, return_complex=True)
        return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

    def mel(self, y: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(self.mel_basis @ self.linear(y.float()), min=1e-5))


def kl_loss(z_p, logs_q, m_p, logs_p, mask):
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    return (kl * mask).sum() / mask.sum()


def feature_loss(fmap_r, fmap_g) -> torch.Tensor:
    return (
        2
        * torch.stack(
            [
                torch.mean(torch.abs(r - g))
                for dr, dg in zip(fmap_r, fmap_g, strict=True)
                for r, g in zip(dr, dg, strict=True)
            ]
        ).sum()
    )


def discriminator_loss(real, fake) -> torch.Tensor:
    return torch.stack(
        [torch.mean((1 - r.float()) ** 2) + torch.mean(g.float() ** 2) for r, g in zip(real, fake, strict=True)]
    ).sum()


def generator_loss(fake) -> torch.Tensor:
    return torch.stack([torch.mean((1 - g.float()) ** 2) for g in fake]).sum()


# ---------------------------------------------------------------- features per clip


def normalize(audio: np.ndarray) -> np.ndarray | None:
    """RVC's level normalization (peak 0.9, mixed 3 : 1 with the original level)."""
    peak = np.abs(audio).max()
    if peak == 0 or peak > 2.5:
        return None
    return audio / peak * (0.9 * 0.75) + 0.25 * audio


PIECE_S, OVERLAP_S = 3.7, 0.3  # RVC cuts training audio into 3.7 s pieces, 0.3 s overlapping


def pieces(audio: np.ndarray, rate: int) -> list[np.ndarray]:
    """RVC's cut: 3.7 s every 3.4 s while more than 4 s remain, then the rest as one piece."""
    out, start = [], 0
    step, size, tail = (int(rate * s) for s in (PIECE_S - OVERLAP_S, PIECE_S, PIECE_S + OVERLAP_S))
    while len(audio) - start > tail:
        out.append(audio[start : start + size])
        start += step
    out.append(audio[start:])
    return out


def clip_features(path: str, contentvec, rmvpe, spectro: Spectrograms, device) -> list[dict]:
    """One clip -> its pieces, each: 40 kHz waveform, linear spectrogram, ContentVec features
    (repeated x2 to 100 per second), coarse + fine pitch, all cut to the same number of frames."""
    y, sr = sf.read(path, dtype="float32")
    y = y.mean(1) if y.ndim == 2 else y
    y40 = librosa.resample(y, orig_sr=sr, target_sr=40000, res_type="soxr_vhq") if sr != 40000 else y
    y40 = signal.lfilter(*HIGHPASS_40K, y40).astype(np.float32)
    out = []
    for piece in pieces(y40, 40000):
        if (features := piece_features(normalize(piece), contentvec, rmvpe, spectro, device)) is not None:
            out.append(features)
    return out


def piece_features(y40: np.ndarray | None, contentvec, rmvpe, spectro: Spectrograms, device) -> dict | None:
    from .convert import coarse_pitch

    if y40 is None:
        return None
    y16 = librosa.resample(y40, orig_sr=40000, target_sr=16000, res_type="soxr_vhq").astype(np.float32)
    feats = contentvec(y16)[0].repeat_interleave(2, dim=0)  # [frames, 768]
    f0 = rmvpe.f0(y16)
    wave = torch.from_numpy(y40.astype(np.float32)).to(device)
    spec = spectro.linear(wave[None])[0]
    n = min(feats.shape[0], len(f0), spec.shape[1], MAX_FRAMES)
    if n < SEGMENT:
        return None
    return {
        "feats": feats[:n].half().cpu(),
        "pitch": torch.from_numpy(coarse_pitch(f0[:n].copy())),
        "f0": torch.from_numpy(f0[:n]).float(),
        "spec": spec[:, :n].half().cpu(),
        "wave": wave[: n * HOP_40K].half().cpu(),
    }


PAD_TO = 50  # frames: batch lengths rounded up, so cuDNN tunes a few shapes, not one per batch


def collate(items: list[dict], device) -> dict:
    n = -(-max(x["feats"].shape[0] for x in items) // PAD_TO) * PAD_TO

    def pad(t, length, dim):
        return F.pad(t, [0, 0] * (t.dim() - 1 - dim) + [0, length - t.shape[dim]])

    return {
        "feats": torch.stack([pad(x["feats"], n, 0) for x in items]).float().to(device),
        "pitch": torch.stack([pad(x["pitch"], n, 0) for x in items]).to(device),
        "f0": torch.stack([pad(x["f0"], n, 0) for x in items]).to(device),
        "spec": torch.stack([pad(x["spec"], n, 1) for x in items]).float().to(device),
        "wave": torch.stack([pad(x["wave"], n * HOP_40K, 0) for x in items]).float().to(device),
        "lengths": torch.tensor([x["feats"].shape[0] for x in items], device=device),
    }


# ---------------------------------------------------------------- the training step


LOSSES = ("disc", "gen", "fm", "mel", "kl")


class Trainer:
    def __init__(
        self, base_g: Path, base_d: Path, device="cuda", lr: float = 1e-4, dtype=torch.bfloat16, compile: bool = False
    ):
        cfg = Config()
        self.device, self.dtype = device, dtype
        torch.backends.cudnn.benchmark = True  # the decoder and discriminator always see 12800-sample windows
        self.g = Synthesizer(cfg)
        with_weight_norm(self.g, norm_layers(self.g))
        state = torch.load(base_g, map_location="cpu", weights_only=True)["model"]
        self.g.load_state_dict({k: v.float() for k, v in state.items()})
        self.d = MultiPeriodDiscriminator()
        state = torch.load(base_d, map_location="cpu", weights_only=True)["model"]
        self.d.load_state_dict({k: v.float() for k, v in state.items()})
        self.g.to(device).train()
        self.d.to(device).train()
        if compile:  # fixed shapes (12800-sample windows): the decoder and each discriminator
            self.g.dec.compile()
            for d in self.d.discriminators:
                d.compile()
        self.opt_g = torch.optim.AdamW(self.g.parameters(), lr, betas=(0.8, 0.99), eps=1e-9)
        self.opt_d = torch.optim.AdamW(self.d.parameters(), lr, betas=(0.8, 0.99), eps=1e-9)
        self.sched_g = torch.optim.lr_scheduler.ExponentialLR(self.opt_g, 0.999875)
        self.sched_d = torch.optim.lr_scheduler.ExponentialLR(self.opt_d, 0.999875)
        self.spectro = Spectrograms(cfg.sample_rate, device)

    def forward_g(self, b: dict):
        g = self.g.emb_g(torch.zeros(len(b["lengths"]), dtype=torch.long, device=self.device)).unsqueeze(-1)
        m_p, logs_p, x_mask = self.g.enc_p(b["feats"], b["pitch"], b["lengths"])
        z, m_q, logs_q, y_mask = self.g.enc_q(b["spec"], b["lengths"], g)
        z_p = self.g.flow(z, y_mask, g)
        # a random 32-frame window of each clip
        starts = (torch.rand(len(b["lengths"]), device=self.device) * (b["lengths"] - SEGMENT + 1)).long()
        idx = starts[:, None] + torch.arange(SEGMENT, device=self.device)[None]
        z_slice = torch.gather(z, 2, idx[:, None].expand(-1, z.shape[1], -1))
        f0_slice = torch.gather(b["f0"], 1, idx)
        y_hat = self.g.dec(z_slice, f0_slice, g)
        wave_idx = (starts * HOP_40K)[:, None] + torch.arange(SEGMENT * HOP_40K, device=self.device)[None]
        wave = torch.gather(b["wave"], 1, wave_idx).unsqueeze(1)
        return y_hat, wave, (z_p, m_p, logs_p, logs_q, y_mask)

    def step(self, b: dict) -> torch.Tensor:
        """One D and one G update; the losses (LOSSES order) as a tensor on the GPU (no sync)."""
        with torch.autocast("cuda", dtype=self.dtype):
            y_hat, wave, (z_p, m_p, logs_p, logs_q, mask) = self.forward_g(b)
            real, fake, _, _ = self.d(wave, y_hat.detach())
        loss_d = discriminator_loss(real, fake)
        self.opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        self.opt_d.step()

        self.d.requires_grad_(False)
        with torch.autocast("cuda", dtype=self.dtype):
            _, fake, fmap_r, fmap_g = self.d(wave, y_hat)
        loss_mel = F.l1_loss(self.spectro.mel(wave.squeeze(1)), self.spectro.mel(y_hat.squeeze(1))) * C_MEL
        loss_kl = kl_loss(z_p.float(), logs_q.float(), m_p.float(), logs_p.float(), mask) * C_KL
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen = generator_loss(fake)
        total = loss_gen + loss_fm + loss_mel + loss_kl
        self.opt_g.zero_grad(set_to_none=True)
        total.backward()
        self.opt_g.step()
        self.d.requires_grad_(True)
        return torch.stack([x.detach().float() for x in (loss_d, loss_gen, loss_fm, loss_mel, loss_kl)])

    def end_epoch(self):
        self.sched_g.step()
        self.sched_d.step()

    def save_voice(self, path: Path, info: str):
        """RVC's voice file: fp16 synthesis weights (no enc_q), config, sr, f0, version."""
        cfg = self.g.cfg
        weights = {k: v.half().cpu() for k, v in plain_state(self.g).items() if not k.startswith("enc_q")}
        config = [getattr(cfg, f) for f in cfg.__dataclass_fields__]
        config = [[list(x) if isinstance(x, tuple) else x for x in v] if isinstance(v, tuple) else v for v in config]
        torch.save({"weight": weights, "config": config, "info": info, "sr": "40k", "f0": 1, "version": "v2"}, path)


def epochs_batches(clips: list, batch: int, seed: int):
    """Clips in batches of similar length (fewer padded frames), shuffled per epoch."""
    rng = random.Random(seed)
    order = sorted(range(len(clips)), key=lambda i: clips[i]["feats"].shape[0])
    groups = [order[i : i + batch] for i in range(0, len(order), batch)]
    rng.shuffle(groups)
    return groups


def steps_per_epoch(n: int, batch: int) -> int:
    return math.ceil(n / batch)
