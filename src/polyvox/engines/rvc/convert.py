"""Voice conversion with an RVC voice: source audio at 16 kHz -> ContentVec features (50 per
second, blended with the voice's nearest training features: the "index") + RMVPE pitch (100
per second, shifted) -> the voice's synthesizer -> audio at the voice's rate.

Long audio is cut at the quietest point near every 38 s, each piece converted with 1 s of
context on both sides (RVC's x_pad / x_query / x_center / x_max = 1 / 6 / 38 / 41 s).
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from .model import Voice
from .pitch import HOP, RATE, RMVPE

PAD_S, QUERY_S, CENTER_S, MAX_S = 1, 6, 38, 41
F0_MIN, F0_MAX = 50.0, 1100.0
MEL_MIN, MEL_MAX = (1127 * np.log(1 + f / 700) for f in (F0_MIN, F0_MAX))
HIGHPASS = signal.butter(N=5, Wn=48, btype="high", fs=RATE)


def clean_f0(f0: np.ndarray, window: int = 15) -> np.ndarray:
    """Octave jumps folded back toward the running median of the voiced frames (~150 ms), then a
    3-frame median. Not in RVC: extractors flip octaves on breathy or reverberant notes."""
    from scipy.ndimage import median_filter

    f0 = f0.copy()
    voiced = f0 > 0
    if voiced.sum() < window:
        return f0
    semis = 12 * np.log2(f0[voiced] / 440.0)
    semis -= 12 * np.round((semis - median_filter(semis, window, mode="nearest")) / 12)
    f0[voiced] = 440.0 * 2 ** (median_filter(semis, 3, mode="nearest") / 12)
    return f0


def coarse_pitch(f0: np.ndarray) -> np.ndarray:
    """f0 (Hz) -> RVC's 255 mel-spaced pitch classes (1 = unvoiced)."""
    mel = 1127 * np.log(1 + f0 / 700)
    voiced = mel > 0
    mel[voiced] = (mel[voiced] - MEL_MIN) * 254 / (MEL_MAX - MEL_MIN) + 1
    return np.rint(np.clip(mel, 1, 255)).astype(np.int64)


class ContentVec:
    """HuBERT-base ContentVec (RVC v2's content features: the last layer, 768 dims)."""

    def __init__(self, folder: Path, device: str = "cuda"):
        from transformers import HubertModel

        self.model = HubertModel.from_pretrained(str(folder)).eval().to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, audio16k: np.ndarray) -> torch.Tensor:  # -> [1, frames, 768]
        x = torch.from_numpy(audio16k).float().to(self.device)[None]
        return self.model(x).last_hidden_state.float()


INDEX_MAX_ROWS, INDEX_CENTERS = 200_000, 10_000  # RVC's: more rows -> k-means to 10k centers


def reduce_features(features: torch.Tensor, iterations: int = 20, seed: int = 0) -> torch.Tensor:
    """A voice's training features for its index: as they are up to 200k rows, else 10k k-means
    centers (RVC does the same with scikit-learn's MiniBatchKMeans)."""
    if features.shape[0] <= INDEX_MAX_ROWS:
        return features
    x = features.float()
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = x[torch.randperm(x.shape[0], generator=g)[:INDEX_CENTERS].to(x.device)].clone()
    for _ in range(iterations):
        assign = torch.cat([torch.cdist(chunk, centers).argmin(1) for chunk in x.split(65536)])
        sums = torch.zeros_like(centers).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=INDEX_CENTERS)[:, None]
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)  # empty clusters stay put
    return centers.to(features.dtype)


class Index:
    """The voice's training features; a frame's 8 nearest (squared L2) are blended by
    inverse-square distance, as RVC's faiss index search does."""

    def __init__(self, features: torch.Tensor):
        self.features = features.float()
        self.norms = (self.features**2).sum(1)

    def blend(self, feats: torch.Tensor, k: int = 8) -> torch.Tensor:  # [frames, 768]
        d = (feats**2).sum(1, keepdim=True) - 2 * feats @ self.features.T + self.norms[None]
        dist, ix = torch.topk(d.clamp_min(0), k, dim=1, largest=False)
        w = 1 / dist.clamp_min(1e-12) ** 2
        w = w / w.sum(1, keepdim=True)
        return (self.features[ix] * w[..., None]).sum(1)


def change_rms(source: np.ndarray, source_rate: int, target: np.ndarray, target_rate: int, rate: float) -> np.ndarray:
    """Give ``target`` the loudness envelope of ``source`` by ``1 - rate`` (rate 1 = unchanged)."""

    def envelope(y, sr, n):
        rms = librosa.feature.rms(y=y, frame_length=sr // 2 * 2, hop_length=sr // 2)
        return F.interpolate(torch.from_numpy(rms).float()[None], size=n, mode="linear")[0, 0]

    r1, r2 = envelope(source, source_rate, len(target)), envelope(target, target_rate, len(target)).clamp_min(1e-6)
    return target * (r1 ** (1 - rate) * r2 ** (rate - 1)).numpy()


@dataclass
class Options:
    shift: int = 0  # semitones
    index_rate: float = 0.75  # how much of the voice's nearest training features replace the source's
    protect: float = 0.33  # < 0.5: unvoiced frames keep this much of the voice's features (breaths, consonants)
    rms_mix: float = 1.0  # 1 = the converted loudness; 0 = the source's envelope
    threshold: float = 0.03  # RMVPE salience below this = unvoiced
    clean_pitch: bool = False  # clean_f0 on the extracted pitch


class Converter:
    def __init__(self, contentvec: ContentVec, rmvpe: RMVPE, device: str = "cuda"):
        self.contentvec, self.rmvpe, self.device = contentvec, rmvpe, device

    def pitch(self, audio: np.ndarray, frames: int, opts: Options) -> tuple[np.ndarray, np.ndarray]:
        x = audio.astype(np.float32)
        scale = np.quantile(np.abs(x), 0.999)
        f0 = self.rmvpe.f0(x / scale if scale > 0 else x, opts.threshold)[:frames]
        if opts.clean_pitch:
            f0 = clean_f0(f0)
        f0 = f0 * 2 ** (opts.shift / 12)
        return coarse_pitch(f0.copy()), f0

    @torch.no_grad()
    def piece(self, voice: Voice, audio: np.ndarray, pitch, f0, index: Index | None, opts: Options) -> np.ndarray:
        feats = self.contentvec(audio)
        feats0 = feats.clone()
        if index is not None and opts.index_rate > 0:
            feats = index.blend(feats[0])[None] * opts.index_rate + (1 - opts.index_rate) * feats
        up = lambda t: F.interpolate(t.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)  # noqa: E731  50 -> 100 per s
        feats, feats0 = up(feats), up(feats0)
        n = min(len(audio) // HOP, feats.shape[1])
        pitch = torch.from_numpy(pitch[:n])[None].to(self.device)
        f0 = torch.from_numpy(f0[:n]).float()[None].to(self.device)
        if opts.protect < 0.5:
            keep = torch.where(f0 > 0, 1.0, opts.protect).unsqueeze(-1)
            feats = feats[:, :n] * keep + feats0[:, :n] * (1 - keep)
        return voice.model.infer(feats[:, :n].float(), pitch, f0)[0].float().cpu().numpy()

    def convert(
        self,
        voice: Voice,
        audio: np.ndarray,
        rate: int,
        index: Index | None = None,
        opts: Options | None = None,
        on_piece=None,
    ) -> np.ndarray:
        """Mono audio at ``rate`` -> the same speech in ``voice``, at ``voice.sample_rate``;
        ``on_piece(done, total)`` after each piece (about 38 s each)."""
        opts = opts or Options()
        audio = (
            librosa.resample(audio.astype(np.float64), orig_sr=rate, target_sr=RATE, res_type="soxr_vhq")
            if rate != RATE
            else audio
        )
        peak = np.abs(audio).max() / 0.95
        if peak > 1:
            audio = audio / peak
        audio = signal.filtfilt(*HIGHPASS, audio)
        t_pad, t_query, t_center, t_max = (s * RATE for s in (PAD_S, QUERY_S, CENTER_S, MAX_S))
        pad_tgt = voice.sample_rate * PAD_S
        cuts = []
        if len(audio) + HOP > t_max:
            padded = np.pad(audio, (HOP // 2, HOP // 2), mode="reflect")
            window_sum = np.convolve(padded, np.ones(HOP), mode="valid")[: len(audio)]
            for t in range(t_center, len(audio), t_center):
                around = np.abs(window_sum[t - t_query : t + t_query])
                cuts.append(t - t_query + int(np.argmin(around)))
        padded = np.pad(audio, (t_pad, t_pad), mode="reflect")
        pitch, f0 = self.pitch(padded, len(padded) // HOP, opts)
        out, s = [], 0
        for i, t in enumerate(cuts):
            t = t // HOP * HOP
            seg = padded[s : t + 2 * t_pad + HOP].astype(np.float32)
            out.append(
                self.piece(
                    voice,
                    seg,
                    pitch[s // HOP : (t + 2 * t_pad) // HOP],
                    f0[s // HOP : (t + 2 * t_pad) // HOP],
                    index,
                    opts,
                )[pad_tgt:-pad_tgt]
            )
            s = t
            if on_piece:
                on_piece(i + 1, len(cuts) + 1)
        out.append(
            self.piece(voice, padded[s:].astype(np.float32), pitch[s // HOP :], f0[s // HOP :], index, opts)[
                pad_tgt:-pad_tgt
            ]
        )
        if on_piece:
            on_piece(len(cuts) + 1, len(cuts) + 1)
        result = np.concatenate(out)
        if opts.rms_mix != 1:
            result = change_rms(audio, RATE, result, voice.sample_rate, opts.rms_mix)
        peak = np.abs(result).max() / 0.99
        return result / peak if peak > 1 else result
