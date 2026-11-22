"""Training an RVC v2 voice (32, 40 or 48 kHz, with pitch) from a dataset's clips: features per
clip (waveform at the voice's rate, linear spectrogram, ContentVec features, RMVPE pitch), then the
VITS-style GAN from RVC's base models (f0G40k / f0D40k, ...): mel L1 x 45 + KL + feature matching +
adversarial (least squares), AdamW 1e-4 (0.8, 0.99), learning rate x 0.999875 per epoch, a random
window of each clip (RATES' segment), bf16 autocast. Saved voices use RVC's file format (fp16
weights, weight_g / weight_v), so RVC's WebUI and Applio load them too.
"""

import functools
import math
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from ...data import audio_io
from .model import Config, Synthesizer


@dataclass(frozen=True)
class Rate:
    """What RVC's v2 configs change per sample rate (every rate runs at 100 frames per second)."""

    n_fft: int  # spectrogram window and FFT size
    n_mels: int
    segment: int  # frames per training window
    upsample_rates: tuple
    upsample_kernel_sizes: tuple


RATES = {
    32000: Rate(1024, 80, 40, (10, 8, 2, 2), (20, 16, 4, 4)),
    40000: Rate(2048, 125, 32, (10, 10, 2, 2), (16, 16, 4, 4)),
    48000: Rate(2048, 128, 36, (12, 10, 2, 2), (24, 20, 4, 4)),
}
FRAMES_PER_S = 100
MAX_FRAMES = 900  # RVC keeps at most 9 s of a clip
C_MEL, C_KL = 45.0, 1.0


def hop(rate: int) -> int:
    return rate // FRAMES_PER_S


def model_config(rate: int) -> Config:
    """The synthesizer of RVC's v2 voices at ``rate`` (its base models' shapes)."""
    r = RATES[rate]
    return Config(
        spec_channels=r.n_fft // 2 + 1,
        segment_size=r.segment,
        upsample_rates=r.upsample_rates,
        upsample_kernel_sizes=r.upsample_kernel_sizes,
        sample_rate=rate,
    )


@functools.cache
def highpass(rate: int):
    return signal.butter(N=5, Wn=48, btype="high", fs=rate)  # RVC's preprocessing filter


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
        r = RATES[sample_rate]
        self.n_fft, self.hop = r.n_fft, hop(sample_rate)
        self.window = torch.hann_window(self.n_fft, device=device)
        mel = librosa.filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=r.n_mels, fmin=0, fmax=None)
        self.mel_basis = torch.from_numpy(mel).float().to(device)

    def linear(self, y: torch.Tensor) -> torch.Tensor:  # [b, samples] -> [b, n_fft / 2 + 1, frames]
        pad = (self.n_fft - self.hop) // 2
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(y, self.n_fft, self.hop, self.n_fft, window=self.window, center=False, return_complex=True)
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


def piece_count(seconds: float, rate: int = 40000) -> int:
    """How many pieces ``pieces`` cuts a clip of ``seconds`` into (for planning steps)."""
    length = int(round(seconds * rate))
    step, tail = int(rate * (PIECE_S - OVERLAP_S)), int(rate * (PIECE_S + OVERLAP_S))
    return 1 + max(0, -(-(length - tail) // step))


def clip_features(path: str, contentvec, rmvpe, rate: int = 40000) -> list[dict]:
    """One clip -> its pieces (piece_features), as RVC's preprocessing makes them: at the voice's
    rate, a 48 Hz high-pass, cut, each piece's level normalized."""
    y, sr = audio_io.read(path)  # m4a / mp3 clips too (PyAV)
    y = y.mean(0)
    y = librosa.resample(y, orig_sr=sr, target_sr=rate, res_type="soxr_vhq") if sr != rate else y
    y = signal.lfilter(*highpass(rate), y).astype(np.float32)
    return [f for piece in pieces(y, rate) if (f := piece_features(normalize(piece), contentvec, rmvpe, rate))]


def piece_features(y: np.ndarray | None, contentvec, rmvpe, rate: int = 40000) -> dict | None:
    """A piece's training inputs, kept compact (on the CPU): ContentVec features at 50 per second
    (float16), coarse and fine pitch and the waveform at ``rate`` (float16) for ``frames`` frames
    at 100 per second (at most 900); the spectrogram is computed per batch (collate). None: too
    short for a training window."""
    from .convert import coarse_pitch

    if y is None:
        return None
    y16 = librosa.resample(y, orig_sr=rate, target_sr=16000, res_type="soxr_vhq").astype(np.float32)
    feats = contentvec(y16)[0]  # [frames / 2, 768]
    f0 = rmvpe.f0(y16)
    n = min(2 * feats.shape[0], len(f0), len(y) // hop(rate), MAX_FRAMES)
    if n < RATES[rate].segment:
        return None
    return {
        "frames": n,
        "feats": feats[: -(-n // 2)].half().cpu(),
        "pitch": torch.from_numpy(coarse_pitch(f0[:n].copy()).astype(np.uint8)),
        "f0": torch.from_numpy(f0[:n]).float(),
        "wave": torch.from_numpy(y[: n * hop(rate)].astype(np.float16)),
    }


PAD_TO = 50  # frames: batch lengths rounded up, so cuDNN tunes a few shapes, not one per batch


def collate(items: list[dict], device, spectro: Spectrograms) -> dict:
    """A batch on ``device``: features repeated to 100 per second, everything padded to a whole
    number of PAD_TO frames, and the linear spectrograms of the waveforms."""
    n = -(-max(x["frames"] for x in items) // PAD_TO) * PAD_TO

    def pad(t, length):
        return F.pad(t, [0, 0] * (t.dim() - 1) + [0, length - t.shape[0]])

    feats = [x["feats"].repeat_interleave(2, dim=0)[: x["frames"]] for x in items]
    wave = torch.stack([pad(x["wave"], n * spectro.hop) for x in items]).to(device).float()
    return {
        "feats": torch.stack([pad(f, n) for f in feats]).to(device).float(),
        "pitch": torch.stack([pad(x["pitch"], n) for x in items]).to(device).long(),
        "f0": torch.stack([pad(x["f0"], n) for x in items]).to(device),
        "wave": wave,
        "spec": spectro.linear(wave),
        "lengths": torch.tensor([x["frames"] for x in items], device=device),
    }


# ---------------------------------------------------------------- the training step


LOSSES = ("disc", "gen", "fm", "mel", "kl")


class Trainer:
    def __init__(
        self,
        base_g: Path,
        base_d: Path,
        device="cuda",
        lr: float = 1e-4,
        dtype=torch.bfloat16,
        compile: bool = False,
        rate: int = 40000,
    ):
        cfg = model_config(rate)
        self.device, self.dtype = device, dtype
        self.rate, self.segment, self.hop = rate, RATES[rate].segment, hop(rate)
        torch.backends.cudnn.benchmark = True  # the decoder and discriminator always see same-size windows
        self.g = Synthesizer(cfg)
        with_weight_norm(self.g, norm_layers(self.g))
        state = torch.load(base_g, map_location="cpu", weights_only=True)["model"]
        self.g.load_state_dict({k: v.float() for k, v in state.items()})
        self.d = MultiPeriodDiscriminator()
        state = torch.load(base_d, map_location="cpu", weights_only=True)["model"]
        self.d.load_state_dict({k: v.float() for k, v in state.items()})
        self.g.to(device).train()
        self.d.to(device).train()
        if compile:  # fixed shapes (one window size): the decoder and each discriminator
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
        # a random window of each clip
        starts = (torch.rand(len(b["lengths"]), device=self.device) * (b["lengths"] - self.segment + 1)).long()
        idx = starts[:, None] + torch.arange(self.segment, device=self.device)[None]
        z_slice = torch.gather(z, 2, idx[:, None].expand(-1, z.shape[1], -1))
        f0_slice = torch.gather(b["f0"], 1, idx)
        y_hat = self.g.dec(z_slice, f0_slice, g)
        wave_idx = (starts * self.hop)[:, None] + torch.arange(self.segment * self.hop, device=self.device)[None]
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

    def state(self) -> dict:
        """Everything a resumed run continues from."""
        return {
            "g": self.g.state_dict(),
            "d": self.d.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "sched_d": self.sched_d.state_dict(),
        }

    def load(self, state: dict) -> None:
        self.g.load_state_dict(state["g"])
        self.d.load_state_dict(state["d"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.sched_g.load_state_dict(state["sched_g"])
        self.sched_d.load_state_dict(state["sched_d"])

    def save_voice(self, path: Path, info: str):
        """RVC's voice file: fp16 synthesis weights (no enc_q), config, sr, f0, version."""
        cfg = self.g.cfg
        weights = {k: v.half().cpu() for k, v in plain_state(self.g).items() if not k.startswith("enc_q")}
        config = [getattr(cfg, f) for f in cfg.__dataclass_fields__]
        config = [[list(x) if isinstance(x, tuple) else x for x in v] if isinstance(v, tuple) else v for v in config]
        sr = f"{self.rate // 1000}k"
        torch.save({"weight": weights, "config": config, "info": info, "sr": sr, "f0": 1, "version": "v2"}, path)


def epochs_batches(clips: list, batch: int, seed: int):
    """Clips in batches of similar length (fewer padded frames), shuffled per epoch."""
    rng = random.Random(seed)
    order = sorted(range(len(clips)), key=lambda i: clips[i]["frames"])
    groups = [order[i : i + batch] for i in range(0, len(order), batch)]
    rng.shuffle(groups)
    return groups


def steps_per_epoch(n: int, batch: int) -> int:
    return math.ceil(n / batch)
