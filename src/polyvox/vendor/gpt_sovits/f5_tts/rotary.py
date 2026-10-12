"""Rotary position embeddings from x_transformers 2.17.10 (lucidrains/x-transformers,
MIT): the two pieces the f5_tts DiT imports, without the package. Pairs are
interleaved ((0, 1), (2, 3), ...) as in that version; the pretrained DiT
depends on it."""

import torch
from torch import nn
from torch.amp import autocast


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward_from_seq_len(self, seq_len):
        return self.forward(torch.arange(seq_len, device=self.inv_freq.device))

    @autocast("cuda", enabled=False)
    def forward(self, t):
        if t.ndim == 1:
            t = t[None]
        freqs = torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)
        return freqs, 1.0


def rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if torch.is_tensor(scale) else scale
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs[:, None]
        if torch.is_tensor(scale):
            scale = scale[:, None]
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim=-1).type(orig_dtype)
