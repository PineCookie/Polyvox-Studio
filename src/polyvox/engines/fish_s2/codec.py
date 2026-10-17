"""Fish Audio S2-Pro's audio codec, written here from its published architecture.

A causal DAC-style codec at 44.1 kHz: a convolutional encoder (hop 512, with a
windowed transformer at its end), two more ×2 downsamplings to ~21.5 frames per
second, a windowed transformer, then residual vector quantization into 10 codes
per frame (one semantic code from 4096, nine residual codes from 1024); decoding
runs the codes back up through a transformer, the upsamplings and the
convolutional decoder. Its weights load from codec.pth (weight norms folded).

Run it in float32: in bfloat16 a round trip loses clearly more of the speaker
(CAM++ similarity of real clips 0.81-0.84 in float32 vs 0.75-0.79 in bfloat16).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import rms_norm, rope_table, rotate

SAMPLE_RATE = 44_100
ENCODER_RATES = (2, 4, 8, 8)  # hop 512
DECODER_RATES = (8, 8, 4, 2)
DOWNSAMPLE = (2, 2)  # in the quantizer: frames of 2048 samples
FRAME = 2048
CODEBOOKS = 9  # residual ones, after the semantic one
SEMANTIC_SIZE, CODEBOOK_SIZE, CODE_DIM = 4096, 1024, 8


class Snake(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class CausalConv(nn.Module):
    """Conv1d padded on the left (and on the right just enough for whole strides)."""

    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, dilation: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel, stride=stride, dilation=dilation, groups=groups)
        self.span = (kernel - 1) * dilation + 1
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        left = self.span - self.stride
        frames = math.ceil((x.shape[-1] - self.span + left) / self.stride + 1)
        right = (frames - 1) * self.stride + self.span - left - x.shape[-1]
        return self.conv(F.pad(x, (left, right)))


class CausalTransConv(nn.Module):
    """ConvTranspose1d with the overhang on the right cut off."""

    def __init__(self, cin: int, cout: int, kernel: int, stride: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout, kernel, stride=stride)
        self.cut = kernel - stride

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        return y[..., : y.shape[-1] - self.cut]


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake(dim), CausalConv(dim, dim, 7, dilation=dilation), Snake(dim), CausalConv(dim, dim, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ConvNeXt(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = CausalConv(dim, dim, 7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(self.dwconv(x).transpose(1, 2))
        y = self.gamma * self.pwconv2(F.gelu(self.pwconv1(y)))
        return x + y.transpose(1, 2)


class Norm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, 1e-5)


class Scale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, table: Tensor, mask: Tensor) -> Tensor:
        b, t, dim = x.shape
        q, k, v = self.wqkv(x).view(b, t, 3, self.heads, dim // self.heads).unbind(2)
        q, k = rotate(q, table), rotate(k, table)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        return self.wo(y.transpose(1, 2).reshape(b, t, dim))


class GatedFeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attention = SelfAttention(dim, heads)
        self.feed_forward = GatedFeedForward(dim, 3 * dim)
        self.attention_norm = Norm(dim)
        self.ffn_norm = Norm(dim)
        self.attention_layer_scale = Scale(dim)
        self.ffn_layer_scale = Scale(dim)

    def forward(self, x: Tensor, table: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attention_layer_scale(self.attention(self.attention_norm(x), table, mask))
        return x + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(x)))


class WindowTransformer(nn.Module):
    """Causal transformer over [B, dim, T], each position seeing the last ``window``."""

    def __init__(self, layers: int, dim: int = 1024, heads: int = 16, window: int = 512):
        super().__init__()
        self.window, self.head_dim = window, dim // heads
        self.layers = nn.ModuleList(TransformerLayer(dim, heads) for _ in range(layers))
        self.norm = Norm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        t = x.shape[1]
        i = torch.arange(t, device=x.device)
        mask = (i[None] <= i[:, None]) & (i[None] > i[:, None] - self.window)
        table = rope_table(t, self.head_dim, 10_000.0, x.device)
        for layer in self.layers:
            x = layer(x, table, mask)
        return self.norm(x).transpose(1, 2)


class Quantizer(nn.Module):
    """One codebook: cosine-nearest code in an 8-dim projection."""

    def __init__(self, size: int, dim: int = 1024):
        super().__init__()
        self.in_proj = nn.Conv1d(dim, CODE_DIM, 1)
        self.out_proj = nn.Conv1d(CODE_DIM, dim, 1)
        self.codebook = nn.Embedding(size, CODE_DIM)

    def encode(self, z: Tensor) -> Tensor:
        e = F.normalize(self.in_proj(z).transpose(1, 2).flatten(0, 1), dim=-1)  # [B * T, 8]
        c = F.normalize(self.codebook.weight, dim=-1)
        distance = e.pow(2).sum(1, keepdim=True) - 2 * e @ c.T + c.pow(2).sum(1)[None]
        return distance.argmin(1).view(z.shape[0], -1)  # [B, T]

    def decode(self, codes: Tensor) -> Tensor:
        return self.out_proj(self.codebook(codes).transpose(1, 2))


class Codebooks(nn.Module):
    def __init__(self, count: int, size: int):
        super().__init__()
        self.quantizers = nn.ModuleList(Quantizer(size) for _ in range(count))


class VectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_quantizer = Codebooks(1, SEMANTIC_SIZE)
        self.quantizer = Codebooks(CODEBOOKS, CODEBOOK_SIZE)
        self.downsample = nn.Sequential(
            *(nn.Sequential(CausalConv(1024, 1024, f, stride=f), ConvNeXt(1024)) for f in DOWNSAMPLE)
        )
        self.upsample = nn.Sequential(
            *(nn.Sequential(CausalTransConv(1024, 1024, f, f), ConvNeXt(1024)) for f in reversed(DOWNSAMPLE))
        )
        self.pre_module = WindowTransformer(8, window=128)
        self.post_module = WindowTransformer(8, window=128)

    def encode(self, z: Tensor) -> Tensor:
        z = self.pre_module(self.downsample(z))
        codes = []
        for q in (*self.semantic_quantizer.quantizers, *self.quantizer.quantizers):
            codes.append(c := q.encode(z))
            z = z - q.decode(c)
        return torch.stack(codes, 1)  # [B, 10, T]

    def decode(self, codes: Tensor) -> Tensor:
        semantic = codes[:, 0].clamp(max=SEMANTIC_SIZE - 1)
        residual = codes[:, 1:].clamp(max=CODEBOOK_SIZE - 1)
        z = self.semantic_quantizer.quantizers[0].decode(semantic)
        for i, q in enumerate(self.quantizer.quantizers):
            z = z + q.decode(residual[:, i])
        return self.upsample(self.post_module(z))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, transformer: bool):
        super().__init__()
        self.block = nn.Sequential(
            *(ResidualUnit(dim // 2, d) for d in (1, 3, 9)),
            Snake(dim // 2),
            CausalConv(dim // 2, dim, 2 * stride, stride=stride),
            *([WindowTransformer(4, dim, dim // 64, window=512)] if transformer else []),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake(cin), CausalTransConv(cin, cout, 2 * stride, stride), *(ResidualUnit(cout, d) for d in (1, 3, 9))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Codec(nn.Module):
    sample_rate = SAMPLE_RATE

    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        dims = [64 * 2 ** (i + 1) for i in range(len(ENCODER_RATES))]  # 128 .. 1024
        self.encoder.block = nn.Sequential(
            CausalConv(1, 64, 7),
            *(EncoderBlock(d, s, i == len(dims) - 1) for i, (d, s) in enumerate(zip(dims, ENCODER_RATES, strict=True))),
            Snake(1024),
            CausalConv(1024, 1024, 3),
        )
        self.quantizer = VectorQuantizer()
        self.decoder = nn.Module()
        chans = [1536 // 2**i for i in range(len(DECODER_RATES) + 1)]  # 1536 .. 96
        self.decoder.model = nn.Sequential(
            CausalConv(1024, 1536, 7),
            *(DecoderBlock(chans[i], chans[i + 1], s) for i, s in enumerate(DECODER_RATES)),
            Snake(chans[-1]),
            CausalConv(chans[-1], 1, 7),
            nn.Tanh(),
        )

    @torch.inference_mode()
    def encode(self, audio: Tensor) -> Tensor:
        """Mono audio at 44.1 kHz [N] -> codes [10, ceil(N / 2048)]."""
        frames = math.ceil(audio.shape[-1] / FRAME)
        x = F.pad(audio.view(1, 1, -1), (0, frames * FRAME - audio.shape[-1]))
        with full_float32():
            return self.quantizer.encode(self.encoder.block(x))[0, :, :frames]

    @torch.inference_mode()
    def decode(self, codes: Tensor) -> Tensor:
        """Codes [10, T] -> mono audio [T * 2048] at 44.1 kHz."""
        with full_float32():
            return self.decoder.model(self.quantizer.decode(codes[None]))[0, 0]


@contextmanager
def full_float32():
    """Without TF32 matmuls / convolutions: their 10-bit mantissas flip nearest-code
    choices (~1 code in 1000 here); sglang-omni encodes references this way too."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def weight_norm(v: Tensor, g: Tensor) -> Tensor:
    """g * v / |v|, the norm over every dimension but the first (torch's weight_norm, dim 0)."""
    return v * (g / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1)))


def folded(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """The checkpoint with its weight norms folded into plain weights and its fixed
    buffers (rotary tables, masks) left out."""
    out = {}
    for key, value in state.items():
        if key.endswith((".freqs_cis", ".causal_mask")) or key.endswith((".original1", ".weight_v")):
            continue
        if key.endswith(".parametrizations.weight.original0"):
            base = key.removesuffix(".parametrizations.weight.original0")
            out[base + ".weight"] = weight_norm(state[base + ".parametrizations.weight.original1"], value)
        elif key.endswith(".weight_g"):
            base = key.removesuffix(".weight_g")
            out[base + ".weight"] = weight_norm(state[base + ".weight_v"], value)
        else:
            out[key] = value
    return out


def load(path: Path, device: str) -> Codec:
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    if any(k.startswith("generator.") for k in state):
        state = {k.removeprefix("generator."): v for k, v in state.items() if k.startswith("generator.")}
    codec = Codec()
    missing, unexpected = codec.load_state_dict(folded(state), strict=False)
    if missing or unexpected:
        raise ValueError(f"S2-Pro codec doesn't fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return codec.to(device).eval().requires_grad_(False)
