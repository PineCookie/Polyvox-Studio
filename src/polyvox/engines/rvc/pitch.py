"""RMVPE pitch extraction (Wei et al. 2023; RVC's rmvpe.pt): 16 kHz audio -> log-mel (128 bands,
hop 160 = 100 frames per second) -> deep U-Net + BiGRU -> 360 pitch classes 20 cents apart ->
f0 in Hz per frame (0 where the salience stays below the threshold)."""

from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_MELS, N_CLASS = 128, 360
HOP, RATE = 160, 16000


class ConvBlockRes(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout, momentum=0.01),
            nn.ReLU(),
        )
        self.shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        return self.conv(x) + (self.shortcut(x) if self.shortcut is not None else x)


class ResBlocks(nn.Module):
    """n ConvBlockRes in a row, then (encoder) a 2 x 2 average pool."""

    def __init__(self, cin: int, cout: int, n: int, pool: bool):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(cin, cout)] + [ConvBlockRes(cout, cout) for _ in range(n - 1)])
        self.pool = nn.AvgPool2d(2) if pool else None

    def forward(self, x):
        for block in self.conv:
            x = block(x)
        return (x, self.pool(x)) if self.pool is not None else x


class Encoder(nn.Module):
    def __init__(self, layers: int = 5, blocks: int = 4, channels: int = 16):
        super().__init__()
        self.bn = nn.BatchNorm2d(1, momentum=0.01)
        self.layers = nn.ModuleList()
        cin = 1
        for _ in range(layers):
            self.layers.append(ResBlocks(cin, channels, blocks, pool=True))
            cin, channels = channels, channels * 2
        self.out_channels = channels  # 512 for the defaults

    def forward(self, x):
        skips = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            skips.append(skip)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, cin: int, cout: int, n: int = 4, blocks: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResBlocks(cin, cout, blocks, False)] + [ResBlocks(cout, cout, blocks, False) for _ in range(n - 1)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, cout: int, blocks: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(cout, momentum=0.01),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList(
            [ConvBlockRes(cout * 2, cout)] + [ConvBlockRes(cout, cout) for _ in range(blocks - 1)]
        )

    def forward(self, x, skip):
        x = torch.cat((self.conv1(x), skip), dim=1)
        for block in self.conv2:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, cin: int, layers: int = 5, blocks: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(DecoderBlock(cin, cin // 2, blocks))
            cin //= 2

    def forward(self, x, skips):
        for layer, skip in zip(self.layers, reversed(skips), strict=True):
            x = layer(x, skip)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.intermediate = Intermediate(self.encoder.out_channels // 2, self.encoder.out_channels)
        self.decoder = Decoder(self.encoder.out_channels)

    def forward(self, x):
        x, skips = self.encoder(x)
        return self.decoder(self.intermediate(x), skips)


class BiGRU(nn.Module):
    def __init__(self, cin: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(cin, hidden, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class E2E(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
        self.cnn = nn.Conv2d(16, 3, 3, padding=1)
        self.fc = nn.Sequential(BiGRU(3 * N_MELS, 256), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):  # [b, mels, frames] -> salience [b, frames, 360]
        x = self.cnn(self.unet(mel.transpose(-1, -2).unsqueeze(1)))
        return self.fc(x.transpose(1, 2).flatten(-2))


class RMVPE:
    def __init__(self, path: Path, device: str = "cuda"):
        self.device = device
        self.model = E2E()
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.eval().to(device)
        basis = librosa.filters.mel(sr=RATE, n_fft=1024, n_mels=N_MELS, fmin=30, fmax=8000, htk=True)
        self.mel_basis = torch.from_numpy(basis).float().to(device)
        self.window = torch.hann_window(1024, device=device)
        cents = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents = np.pad(cents, (4, 4))

    def mel(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(audio, 1024, HOP, 1024, window=self.window, center=True, return_complex=True).abs()
        return torch.log(torch.clamp(self.mel_basis @ spec, min=1e-5))

    @torch.inference_mode()
    def salience(self, audio: np.ndarray) -> np.ndarray:
        mel = self.mel(torch.from_numpy(audio).float().to(self.device)[None])
        n = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n - 1) // 32 + 1) - n), mode="reflect")  # the U-Net halves 5 times
        return self.model(mel)[0, :n].float().cpu().numpy()

    def decode(self, salience: np.ndarray, threshold: float) -> np.ndarray:
        """Weighted mean of the 9 classes around the peak, in cents -> Hz."""
        center = salience.argmax(axis=1) + 4
        padded = np.pad(salience, ((0, 0), (4, 4)))
        idx = center[:, None] + np.arange(-4, 5)[None]
        local = padded[np.arange(len(padded))[:, None], idx]
        cents = (local * self.cents[idx]).sum(1) / local.sum(1)
        f0 = 10 * 2 ** (cents / 1200)
        f0[salience.max(axis=1) <= threshold] = 0
        return f0

    def f0(self, audio: np.ndarray, threshold: float = 0.03, chunk_s: float = 10.0) -> np.ndarray:
        """16 kHz audio -> f0 per 10 ms frame (len(audio) // 160 + 1 frames). Long audio goes in
        windows of ``chunk_s`` with 1 s of context each side (only the middle is kept)."""
        frames = len(audio) // HOP + 1
        chunk = int(chunk_s * RATE) // HOP
        if len(audio) <= chunk * HOP:
            return self.decode(self.salience(audio), threshold)
        out = np.empty(frames, dtype=np.float64)
        context = RATE // HOP
        for start in range(0, frames - 1, chunk):
            end = frames if start + chunk >= frames - 1 else start + chunk
            left = max(0, start - context)
            right = min(len(audio), (end + context) * HOP)
            values = self.decode(self.salience(audio[left * HOP : right]), threshold)
            out[start:end] = values[start - left : start - left + end - start]
        return out
