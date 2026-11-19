"""RVC v2's synthesizer (VITS without the duration path): content features + coarse pitch ->
prior (enc_p) -> flow (reversed) -> neural-source-filter HiFi-GAN (dec), with the speaker as
an embedding (emb_g). The posterior encoder (enc_q) is used only in training.

Parameter names follow RVC's checkpoints, so its voices and base models load as they are.
Weight norm: the checkpoints store ``weight_g`` / ``weight_v`` pairs; ``load_voice`` folds
them into plain weights for synthesis.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

LRELU = 0.1


@dataclass(frozen=True)
class Config:
    """RVC's model config (the list stored in a voice file, in that order)."""

    spec_channels: int = 1025
    segment_size: int = 32
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.0
    resblock: str = "1"
    resblock_kernel_sizes: tuple = (3, 7, 11)
    resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates: tuple = (10, 10, 2, 2)
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: tuple = (16, 16, 4, 4)
    spk_embed_dim: int = 109
    gin_channels: int = 256
    sample_rate: int = 40000

    @classmethod
    def from_list(cls, values: list) -> "Config":
        names = list(cls.__dataclass_fields__)
        fixed = [tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v for v in values]
        return cls(**dict(zip(names, fixed, strict=True)))

    @property
    def hop(self) -> int:  # output samples per feature frame (100 frames per second)
        return math.prod(self.upsample_rates)


def sequence_mask(lengths: torch.Tensor, length: int) -> torch.Tensor:
    return torch.arange(length, device=lengths.device)[None] < lengths[:, None]


class LayerNorm(nn.Module):
    """Layer norm over the channels of [batch, channels, time]."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return F.layer_norm(x.transpose(1, -1), (x.size(1),), self.gamma, self.beta, self.eps).transpose(1, -1)


class RelativeAttention(nn.Module):
    """Self-attention with relative position embeddings (window 10, shared by the heads)."""

    def __init__(self, channels: int, n_heads: int, window: int = 10):
        super().__init__()
        self.n_heads, self.k = n_heads, channels // n_heads
        self.window = window
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        std = self.k**-0.5
        self.emb_rel_k = nn.Parameter(torch.randn(1, 2 * window + 1, self.k) * std)
        self.emb_rel_v = nn.Parameter(torch.randn(1, 2 * window + 1, self.k) * std)

    def relative(self, emb: torch.Tensor, length: int) -> torch.Tensor:
        """Embeddings for offsets -(length - 1) .. length - 1 (zero beyond the window)."""
        pad = max(length - (self.window + 1), 0)
        start = max((self.window + 1) - length, 0)
        if pad:
            emb = F.pad(emb, (0, 0, pad, pad))
        return emb[:, start : start + 2 * length - 1]

    def forward(self, x, mask):
        b, c, t = x.shape
        q, k, v = (
            conv(x).view(b, self.n_heads, self.k, t).transpose(2, 3) for conv in (self.conv_q, self.conv_k, self.conv_v)
        )
        q = q / math.sqrt(self.k)
        scores = q @ k.transpose(-2, -1)
        # relative logits [b, h, t, 2t - 1] -> absolute positions [b, h, t, t]
        rel = q @ self.relative(self.emb_rel_k, t).unsqueeze(0).transpose(-2, -1)
        rel = F.pad(rel, (0, 1)).view(b, self.n_heads, t * 2 * t)
        rel = F.pad(rel, (0, t - 1)).view(b, self.n_heads, t + 1, 2 * t - 1)[:, :, :t, t - 1 :]
        scores = (scores + rel).masked_fill(mask == 0, -1e4)
        p = torch.softmax(scores, dim=-1)
        out = p @ v
        # absolute weights [b, h, t, t] -> relative [b, h, t, 2t - 1]
        w = F.pad(p, (0, t - 1)).view(b, self.n_heads, t * t + t * (t - 1))
        w = F.pad(w, (t, 0)).view(b, self.n_heads, t, 2 * t)[:, :, :, 1:]
        out = out + w @ self.relative(self.emb_rel_v, t).unsqueeze(0)
        return self.conv_o(out.transpose(2, 3).reshape(b, c, t))


class FFN(nn.Module):
    def __init__(self, channels: int, filter_channels: int, kernel_size: int):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.conv_1 = nn.Conv1d(channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, channels, kernel_size)

    def forward(self, x, x_mask):
        x = torch.relu(self.conv_1(F.pad(x * x_mask, (self.pad, self.pad))))
        return self.conv_2(F.pad(x * x_mask, (self.pad, self.pad))) * x_mask


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        h, n = cfg.hidden_channels, cfg.n_layers
        self.drop = nn.Dropout(cfg.p_dropout)
        self.attn_layers = nn.ModuleList(RelativeAttention(h, cfg.n_heads) for _ in range(n))
        self.norm_layers_1 = nn.ModuleList(LayerNorm(h) for _ in range(n))
        self.ffn_layers = nn.ModuleList(FFN(h, cfg.filter_channels, cfg.kernel_size) for _ in range(n))
        self.norm_layers_2 = nn.ModuleList(LayerNorm(h) for _ in range(n))

    def forward(self, x, x_mask):
        mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, n1, ffn, n2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2, strict=True
        ):
            x = n1(x + self.drop(attn(x, mask)))
            x = n2(x + self.drop(ffn(x, x_mask)))
        return x * x_mask


class TextEncoder(nn.Module):
    """Prior: content features (768) + coarse pitch (1..255) -> mean and log std."""

    def __init__(self, cfg: Config, feature_dim: int = 768):
        super().__init__()
        self.out_channels = cfg.inter_channels
        self.scale = math.sqrt(cfg.hidden_channels)
        self.emb_phone = nn.Linear(feature_dim, cfg.hidden_channels)
        self.emb_pitch = nn.Embedding(256, cfg.hidden_channels)
        self.encoder = Encoder(cfg)
        self.proj = nn.Conv1d(cfg.hidden_channels, cfg.inter_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        x = F.leaky_relu((self.emb_phone(phone) + self.emb_pitch(pitch)) * self.scale, LRELU).transpose(1, -1)
        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        stats = self.proj(self.encoder(x, x_mask)) * x_mask
        m, logs = stats.split(self.out_channels, dim=1)
        return m, logs, x_mask


class WaveNet(nn.Module):
    """Non-causal WaveNet with global conditioning (gated tanh / sigmoid units)."""

    def __init__(self, hidden: int, kernel_size: int, dilation_rate: int, n_layers: int, gin_channels: int):
        super().__init__()
        self.hidden = hidden
        self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden * n_layers, 1)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        for i in range(n_layers):
            d = dilation_rate**i
            self.in_layers.append(
                nn.Conv1d(hidden, 2 * hidden, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2)
            )
            self.res_skip_layers.append(nn.Conv1d(hidden, hidden if i == n_layers - 1 else 2 * hidden, 1))

    def forward(self, x, x_mask, g):
        out = torch.zeros_like(x)
        g = self.cond_layer(g)
        h = self.hidden
        last = len(self.in_layers) - 1
        for i, (inp, res_skip) in enumerate(zip(self.in_layers, self.res_skip_layers, strict=True)):
            a = inp(x) + g[:, i * 2 * h : (i + 1) * 2 * h]
            acts = res_skip(torch.tanh(a[:, :h]) * torch.sigmoid(a[:, h:]))
            if i < last:
                x = (x + acts[:, :h]) * x_mask
                out = out + acts[:, h:]
            else:
                out = out + acts
        return out * x_mask


class CouplingLayer(nn.Module):
    """Mean-only affine coupling: shifts the second half by a function of the first."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.half = cfg.inter_channels // 2
        self.pre = nn.Conv1d(self.half, cfg.hidden_channels, 1)
        self.enc = WaveNet(cfg.hidden_channels, 5, 1, 3, cfg.gin_channels)
        self.post = nn.Conv1d(cfg.hidden_channels, self.half, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x, x_mask, g, reverse: bool):
        x0, x1 = x.split(self.half, 1)
        m = self.post(self.enc(self.pre(x0) * x_mask, x_mask, g)) * x_mask
        x1 = (x1 - m) * x_mask if reverse else m + x1 * x_mask
        return torch.cat([x0, x1], 1)


class Flip(nn.Module):
    def forward(self, x, *_, **__):
        return torch.flip(x, [1])


class Flow(nn.Module):
    def __init__(self, cfg: Config, n_flows: int = 4):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.extend([CouplingLayer(cfg), Flip()])

    def forward(self, x, x_mask, g, reverse: bool = False):
        for flow in reversed(self.flows) if reverse else self.flows:
            x = flow(x, x_mask, g, reverse=reverse) if isinstance(flow, CouplingLayer) else flow(x)
        return x


class PosteriorEncoder(nn.Module):
    """Training only: linear spectrogram -> latent z (sampled)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.out_channels = cfg.inter_channels
        self.pre = nn.Conv1d(cfg.spec_channels, cfg.hidden_channels, 1)
        self.enc = WaveNet(cfg.hidden_channels, 5, 1, 16, cfg.gin_channels)
        self.proj = nn.Conv1d(cfg.hidden_channels, cfg.inter_channels * 2, 1)

    def forward(self, spec, lengths, g):
        x_mask = sequence_mask(lengths, spec.size(2)).unsqueeze(1).to(spec.dtype)
        stats = self.proj(self.enc(self.pre(spec) * x_mask, x_mask, g)) * x_mask
        m, logs = stats.split(self.out_channels, dim=1)
        return (m + torch.randn_like(m) * torch.exp(logs)) * x_mask, m, logs, x_mask


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple):
        super().__init__()

        def conv(d):
            return nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2)

        self.convs1 = nn.ModuleList(conv(d) for d in dilations)
        self.convs2 = nn.ModuleList(conv(1) for _ in dilations)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
        return x


def sine_source(f0: torch.Tensor, hop: int, sample_rate: int, amp: float = 0.1, noise_std: float = 0.003):
    """The excitation: a sine at f0 (voiced frames) plus noise, at the output rate.
    f0 [batch, frames] in Hz (0 = unvoiced) -> [batch, frames * hop, 1]."""
    b, n = f0.shape
    f0 = f0.unsqueeze(-1)
    phase = (f0 / sample_rate) * torch.arange(1, hop + 1, dtype=f0.dtype, device=f0.device)  # [b, n, hop]
    # carry each frame's end phase into the next (wrapped to keep float32 precise)
    carry = (torch.fmod(phase[:, :-1, -1:] + 0.5, 1.0) - 0.5).cumsum(dim=1).fmod(1.0)
    phase = phase + F.pad(carry, (0, 0, 1, 0))
    sine = torch.sin(2 * math.pi * phase.reshape(b, -1, 1)) * amp
    voiced = (f0 > 0).to(f0.dtype).repeat_interleave(hop, dim=1)  # [b, n * hop, 1]
    noise = (voiced * noise_std + (1 - voiced) * amp / 3) * torch.randn_like(sine)
    return sine * voiced + noise


class SourceModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_linear = nn.Linear(1, 1)

    def forward(self, f0, hop, sample_rate):
        with torch.no_grad():
            s = sine_source(f0, hop, sample_rate)
        return torch.tanh(self.l_linear(s.to(self.l_linear.weight.dtype)))


class Generator(nn.Module):
    """HiFi-GAN with a neural-source-filter excitation added after each upsampling."""

    def __init__(self, cfg: Config):
        super().__init__()
        rates, kernels, ch0 = cfg.upsample_rates, cfg.upsample_kernel_sizes, cfg.upsample_initial_channel
        self.sample_rate, self.hop = cfg.sample_rate, cfg.hop
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.m_source = SourceModule()
        self.conv_pre = nn.Conv1d(cfg.inter_channels, ch0, 7, 1, padding=3)
        self.cond = nn.Conv1d(cfg.gin_channels, ch0, 1)
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (u, k) in enumerate(zip(rates, kernels, strict=True)):
            out = ch0 // 2 ** (i + 1)
            self.ups.append(nn.ConvTranspose1d(ch0 // 2**i, out, k, u, padding=(k - u) // 2))
            stride = math.prod(rates[i + 1 :])
            if stride > 1:
                self.noise_convs.append(nn.Conv1d(1, out, stride * 2, stride=stride, padding=stride // 2))
            else:
                self.noise_convs.append(nn.Conv1d(1, out, 1))
            for kr, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes, strict=True):
                self.resblocks.append(ResBlock(out, kr, d))
        self.conv_post = nn.Conv1d(out, 1, 7, 1, padding=3, bias=False)

    def forward(self, x, f0, g):
        source = self.m_source(f0, self.hop, self.sample_rate).transpose(1, 2)  # [b, 1, samples]
        x = self.conv_pre(x) + self.cond(g)
        for i, (up, noise) in enumerate(zip(self.ups, self.noise_convs, strict=True)):
            x = up(F.leaky_relu(x, LRELU)) + noise(source)
            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x)))


class Synthesizer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.enc_p = TextEncoder(cfg)
        self.dec = Generator(cfg)
        self.enc_q = PosteriorEncoder(cfg)
        self.flow = Flow(cfg)
        self.emb_g = nn.Embedding(cfg.spk_embed_dim, cfg.gin_channels)

    @torch.no_grad()
    def infer(self, features, pitch, f0, speaker: int = 0, noise: float = 0.66666):
        """features [b, frames, 768] (100 per second), pitch [b, frames] coarse 1..255,
        f0 [b, frames] Hz -> audio [b, frames * hop]."""
        lengths = torch.full((features.shape[0],), features.shape[1], device=features.device)
        g = self.emb_g(torch.full((features.shape[0],), speaker, device=features.device)).unsqueeze(-1)
        m, logs, x_mask = self.enc_p(features, pitch, lengths)
        z_p = (m + torch.exp(logs) * torch.randn_like(m) * noise) * x_mask
        z = self.flow(z_p, x_mask, g, reverse=True)
        return self.dec(z * x_mask, f0, g)[:, 0]


def fold_weight_norm(state: dict) -> dict:
    """``weight_g`` / ``weight_v`` pairs -> ``weight`` (torch's weight_norm over dim 0)."""
    out = {}
    for key, value in state.items():
        if key.endswith(".weight_g"):
            continue
        if key.endswith(".weight_v"):
            base = key[: -len("weight_v")]
            g = state[base + "weight_g"].float()
            v = value.float()
            norm = v.norm(dim=tuple(range(1, v.dim())), keepdim=True)
            out[base + "weight"] = g * v / norm
        else:
            out[key] = value
    return out


@dataclass
class Voice:
    model: Synthesizer
    sample_rate: int
    version: str
    info: str


def load_voice(path: Path, device: str = "cuda") -> Voice:
    """An RVC v2 voice file with pitch (``f0`` 1), as RVC's WebUI saves them."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("version", "v1") != "v2" or not ck.get("f0", 1):
        raise ValueError(f"{Path(path).name}: only RVC v2 voices with pitch are supported.")
    cfg = Config.from_list(ck["config"])
    state = fold_weight_norm(ck["weight"])
    cfg = Config(**{**cfg.__dict__, "spk_embed_dim": state["emb_g.weight"].shape[0]})
    model = Synthesizer(cfg)
    del model.enc_q  # synthesis only
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"{Path(path).name}: not an RVC v2 voice ({len(missing)} missing, {len(unexpected)} unexpected keys)"
        )
    return Voice(model.eval().to(device), cfg.sample_rate, ck["version"], str(ck.get("info", "")))
