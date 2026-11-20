"""Lead and backing vocals apart (covers: the harmonies kept as sung, or converted on their own).

UVR's MDX-Net Karaoke 2 (``UVR_MDXNET_KARA_2.onnx``, MIT, Ultimate Vocal Remover), run with
onnxruntime (on the CPU, faster than real time). Our wrapper around the network: 44.1 kHz stereo,
STFT n_fft 5120 / hop 1024 (periodic Hann, centered), the first 2048 bins in as [batch, 4 (left
re, left im, right re, right im), 2048, 256], chunks of 256 frames with n_fft / 2 dropped at each
edge, the input peak-normalized. On a vocal stem the network gives the part that isn't the lead
voice (the backing); the lead is the rest: vocals - 1.065 x backing.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

RATE = 44_100
N_FFT, HOP, DIM_F, DIM_T = 5120, 1024, 2048, 256
COMPENSATE = 1.065  # UVR's for this model: the network's output runs a little quiet


def load_session(path: str):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


SessionLoader = Callable[[str], object]


class Karaoke:
    """One loaded model (an onnxruntime session), kept for later songs; unload() frees it."""

    def __init__(self, loader: SessionLoader | None = None):
        self._loader = loader  # None: load_session, looked up when loading (tests replace it)
        self._lock = threading.Lock()
        self._session = None  # (path, session)

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def unload(self) -> None:
        with self._lock:
            self._session = None

    def split(
        self, model: str, vocals: np.ndarray, on_progress: Callable[[int, int], None] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(2, n) vocals at 44.1 kHz -> (lead, backing), both (2, n) float32."""
        import torch

        with self._lock:
            if self._session is None or self._session[0] != model:
                self._session = (model, (self._loader or load_session)(model))
            session = self._session[1]
            name = session.get_inputs()[0].name
            peak = float(np.abs(vocals).max()) or 1.0
            n = vocals.shape[1]
            trim = N_FFT // 2
            chunk = HOP * (DIM_T - 1)
            gen = chunk - 2 * trim
            pad = gen - n % gen
            padded = np.concatenate(
                [np.zeros((2, trim)), vocals / peak, np.zeros((2, pad)), np.zeros((2, trim))], 1
            ).astype(np.float32)
            window = torch.hann_window(N_FFT, periodic=True)
            starts = range(0, n + pad, gen)
            pieces = []
            for i, start in enumerate(starts):
                x = torch.from_numpy(padded[:, start : start + chunk])
                spec = torch.stft(x, N_FFT, HOP, window=window, center=True, return_complex=True)
                spec = torch.view_as_real(spec).permute(0, 3, 1, 2).reshape(1, 4, N_FFT // 2 + 1, DIM_T)
                y = torch.from_numpy(session.run(None, {name: spec[:, :, :DIM_F].numpy()})[0])
                y = torch.cat([y, torch.zeros(1, 4, N_FFT // 2 + 1 - DIM_F, DIM_T)], 2)
                y = y.reshape(2, 2, N_FFT // 2 + 1, DIM_T).permute(0, 2, 3, 1).contiguous()
                wave = torch.istft(torch.view_as_complex(y), N_FFT, HOP, window=window, center=True, length=chunk)
                pieces.append(wave[:, trim:-trim].numpy())
                if on_progress:
                    on_progress(i + 1, len(starts))
            backing = (np.concatenate(pieces, 1)[:, :n] * peak).astype(np.float32)
            lead = (vocals - COMPENSATE * backing).astype(np.float32)
            return lead, backing
