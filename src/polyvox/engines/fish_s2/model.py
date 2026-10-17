"""Fish Audio S2-Pro's language model, written here from its published architecture.

A "dual autoregressive" model over audio frames of 10 codebooks (~21.5 per second):
the slow transformer (Qwen3-shaped, 36 layers) reads the text and the frames so far
and predicts each frame's first, semantic code as a token of its vocabulary; the
fast transformer (4 layers) then predicts the other 9 codes of that frame, one
after another, starting from the slow transformer's last (normalized) hidden state.

Matches the checkpoint's layout and numerics: rotary embeddings on interleaved
pairs with the angle table rounded to bfloat16, per-head RMS norms on queries and
keys (slow only), output head tied to the token embeddings, and at audio positions
the token embedding plus the sum of the frame's codebook embeddings, scaled by
1/sqrt(codebooks + 1).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class Shape:
    dim: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    hidden: int  # feed-forward width
    qk_norm: bool
    eps: float
    rope_base: float


@dataclass(frozen=True)
class Config:
    slow: Shape
    fast: Shape
    vocab_size: int
    codebooks: int  # 10: the semantic code and 9 residual ones
    codebook_size: int  # 4096 (the residual codebooks use the first 1024)
    semantic_begin: int  # token id of <|semantic:0|>
    semantic_end: int  # token id of <|semantic:4095|>

    @staticmethod
    def read(model_dir: Path) -> Config:
        data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        text, audio = data["text_config"], data["audio_decoder_config"]

        def shape(c: dict) -> Shape:
            return Shape(
                dim=c["dim"],
                layers=c["n_layer"],
                heads=c["n_head"],
                kv_heads=c.get("n_local_heads", c["n_head"]),
                head_dim=c["head_dim"],
                hidden=c["intermediate_size"],
                qk_norm=bool(c.get("attention_qk_norm")),
                eps=c["norm_eps"],
                rope_base=c["rope_base"],
            )

        return Config(
            slow=shape(text),
            fast=shape(audio),
            vocab_size=text["vocab_size"],
            codebooks=audio["num_codebooks"],
            codebook_size=audio["vocab_size"],
            semantic_begin=data["semantic_start_token_id"],
            semantic_end=data["semantic_end_token_id"],
        )


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Normalized in float32, scaled in the input's precision (as the checkpoint was trained)."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * weight


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


def rope_table(positions: int, head_dim: int, base: float, device=None) -> Tensor:
    """[positions, head_dim / 2, 2] cos / sin of each pair's angle, rounded to bfloat16
    (computed on the CPU, as the checkpoint's own code does)."""
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(positions).float(), inv)
    return torch.stack([angles.cos(), angles.sin()], -1).to(torch.bfloat16).to(device)


def rotate(x: Tensor, table: Tensor) -> Tensor:
    """Rotary embedding on interleaved pairs (x[2i], x[2i+1]); x: [B, T, H, D], table: [T, D/2, 2]."""
    pairs = x.float().unflatten(-1, (-1, 2))
    cos, sin = table.float()[None, :, None, :, 0], table.float()[None, :, None, :, 1]
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack([a * cos - b * sin, b * cos + a * sin], -1).flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, s: Shape, plain: bool = False):
        super().__init__()
        self.s, self.plain = s, plain
        self.wqkv = nn.Linear(s.dim, (s.heads + 2 * s.kv_heads) * s.head_dim, bias=False)
        self.wo = nn.Linear(s.heads * s.head_dim, s.dim, bias=False)
        if s.qk_norm:
            self.q_norm = nn.RMSNorm(s.head_dim, s.eps)
            self.k_norm = nn.RMSNorm(s.head_dim, s.eps)
        self.cache: tuple[Tensor, Tensor] | None = None  # [1, kv_heads, capacity, head_dim] each

    def forward(self, x: Tensor, table: Tensor, start: int) -> Tensor:
        """x: [1, T, dim] at positions start .. start + T - 1, attending to the cache's prefix."""
        s, (b, t, _) = self.s, x.shape
        q, k, v = self.wqkv(x).split([s.heads * s.head_dim, s.kv_heads * s.head_dim, s.kv_heads * s.head_dim], -1)
        q = q.view(b, t, s.heads, s.head_dim)
        k = k.view(b, t, s.kv_heads, s.head_dim)
        v = v.view(b, t, s.kv_heads, s.head_dim)
        if s.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = rotate(q, table), rotate(k, table)
        q, k, v = (y.transpose(1, 2) for y in (q, k, v))
        ck, cv = self.cache
        ck[:, :, start : start + t] = k
        cv[:, :, start : start + t] = v
        k, v = ck[:, :, : start + t], cv[:, :, : start + t]
        groups = s.heads // s.kv_heads
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        if self.plain:  # the fast transformer's arithmetic as trained (bfloat16 matmul and softmax)
            y = torch.softmax((q @ k.transpose(-2, -1)) * (1 / math.sqrt(s.head_dim)), -1) @ v
        elif t == 1:  # one new position sees the whole prefix
            y = F.scaled_dot_product_attention(q, k, v)
        else:  # positions start .. start + t - 1 each see the prefix and themselves
            mask = torch.ones(t, start + t, dtype=torch.bool, device=x.device).tril(start)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(y.transpose(1, 2).reshape(b, t, s.heads * s.head_dim))


class FeedForward(nn.Module):
    def __init__(self, s: Shape):
        super().__init__()
        self.w1 = nn.Linear(s.dim, s.hidden, bias=False)
        self.w3 = nn.Linear(s.dim, s.hidden, bias=False)
        self.w2 = nn.Linear(s.hidden, s.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, s: Shape, plain: bool = False):
        super().__init__()
        self.attention = Attention(s, plain)
        self.feed_forward = FeedForward(s)
        self.attention_norm = RMSNorm(s.dim, s.eps)
        self.ffn_norm = RMSNorm(s.dim, s.eps)

    def forward(self, x: Tensor, table: Tensor, start: int) -> Tensor:
        h = x + self.attention(self.attention_norm(x), table, start)
        return h + self.feed_forward(self.ffn_norm(h))


class DualAR(nn.Module):
    """Parameter names follow the checkpoint (see load): slow = layers / norm /
    embeddings / codebook_embeddings; fast = fast_*."""

    def __init__(self, config: Config):
        super().__init__()
        c = config
        self.config = c
        self.embeddings = nn.Embedding(c.vocab_size, c.slow.dim)
        self.codebook_embeddings = nn.Embedding(c.codebook_size * c.codebooks, c.slow.dim)
        self.layers = nn.ModuleList(Block(c.slow) for _ in range(c.slow.layers))
        self.norm = RMSNorm(c.slow.dim, c.slow.eps)
        self.fast_embeddings = nn.Embedding(c.codebook_size, c.fast.dim)
        self.fast_layers = nn.ModuleList(Block(c.fast, plain=True) for _ in range(c.fast.layers))
        self.fast_norm = RMSNorm(c.fast.dim, c.fast.eps)
        self.fast_output = nn.Linear(c.fast.dim, c.codebook_size, bias=False)
        self.slow_table: Tensor | None = None
        self.fast_table: Tensor | None = None
        self.capacity = 0

    def allocate(self, capacity: int) -> None:
        """KV caches for ``capacity`` positions (slow) and one frame (fast)."""
        c, p = self.config, self.embeddings.weight
        if capacity <= self.capacity:
            return
        self.capacity = capacity
        for layer in self.layers:
            shape = (1, c.slow.kv_heads, capacity, c.slow.head_dim)
            layer.attention.cache = (p.new_zeros(shape), p.new_zeros(shape))
        for layer in self.fast_layers:
            shape = (1, c.fast.kv_heads, c.codebooks, c.fast.head_dim)
            layer.attention.cache = (p.new_zeros(shape), p.new_zeros(shape))
        self.slow_table = rope_table(capacity, c.slow.head_dim, c.slow.rope_base, p.device)
        self.fast_table = rope_table(c.codebooks, c.fast.head_dim, c.fast.rope_base, p.device)

    def slow_parts(self) -> list[nn.Module]:
        """The slow transformer and the embeddings it reads (~7.7 GB): training only the fast
        transformer from cached slow states needs none of them on the GPU."""
        return [self.layers, self.norm, self.embeddings, self.codebook_embeddings]

    def release(self) -> None:
        """Drop the KV caches and angle tables (allocate makes them again)."""
        for layer in (*self.layers, *self.fast_layers):
            layer.attention.cache = None
        self.slow_table = self.fast_table = None
        self.capacity = 0

    def embed(self, tokens: Tensor) -> Tensor:
        """tokens: [codebooks + 1, T] (row 0: token ids; rows 1..: a frame's codes where
        row 0 holds a semantic token) -> [1, T, dim]."""
        c = self.config
        x = self.embeddings(tokens[0])
        audio = (tokens[0] >= c.semantic_begin) & (tokens[0] <= c.semantic_end)
        if bool(audio.any()):
            offsets = torch.arange(c.codebooks, device=tokens.device)[:, None] * c.codebook_size
            codes = self.codebook_embeddings(tokens[1:] + offsets).sum(0)
            x = torch.where(audio[:, None], (x + codes) / math.sqrt(c.codebooks + 1), x)
        return x[None]

    def slow(self, tokens: Tensor, start: int) -> tuple[Tensor, Tensor]:
        """Run the slow transformer over ``tokens`` (positions start ..); returns the last
        position's token logits [vocab] and normalized hidden state [dim]."""
        x = self.embed(tokens)
        table = self.slow_table[start : start + x.shape[1]]
        for layer in self.layers:
            x = layer(x, table, start)
        h = self.norm(x[:, -1:])
        return F.linear(h, self.embeddings.weight)[0, 0], h[0, 0]

    def fast(self, x: Tensor, position: int) -> Tensor:
        """One fast-transformer step: x [dim] at ``position`` (0 = the slow hidden state,
        i = the embedding of code i - 1) -> logits of code ``position`` [codebook_size]."""
        x = x.view(1, 1, -1)
        table = self.fast_table[position : position + 1]
        for layer in self.fast_layers:
            x = layer(x, table, position)
        return self.fast_output(self.fast_norm(x))[0, 0]


def checkpoint_name(key: str) -> str | None:
    """Our parameter name for a checkpoint key (None: not ours)."""
    if key.startswith("text_model.model."):
        return key.removeprefix("text_model.model.")
    if key.startswith("audio_decoder.codebook_embeddings."):
        return key.removeprefix("audio_decoder.")
    if key.startswith("audio_decoder."):
        return "fast_" + key.removeprefix("audio_decoder.")
    return None


def load(model_dir: Path, device: str, dtype: torch.dtype = torch.bfloat16) -> DualAR:
    """The model from its sharded safetensors, straight onto ``device`` (no float32 copy)."""
    from safetensors import safe_open

    config = Config.read(model_dir)
    with torch.device("meta"):
        model = DualAR(config)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weights = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(model_dir / shard), framework="pt", device=str(device)) as f:
            for key in f.keys():  # noqa: SIM118 (safe_open has no iteration)
                if name := checkpoint_name(key):
                    weights[name] = f.get_tensor(key).to(dtype)
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"S2-Pro checkpoint doesn't fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.eval().requires_grad_(False)
