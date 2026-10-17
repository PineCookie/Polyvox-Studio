"""Generating S2-Pro's audio frames: the slow transformer picks each frame's
semantic token, the fast one its other codes.

Sampling as the model's authors do it: top-p and top-k chosen on the untempered
distribution, then the temperature, then an exponential-race draw. Against
loops, "repetition-aware sampling": when the semantic token drawn was already
among the last 10, it is drawn again at temperature 1.0 / top-p 0.9.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import DualAR

RAS_WINDOW, RAS_TEMPERATURE, RAS_TOP_P = 10, 1.0, 0.9


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30


DEFAULT_SAMPLING = Sampling()  # the authors' API defaults


def draw(logits: Tensor, temperature: float, top_p: float, top_k: int, generator=None) -> int:
    """One index from logits [N] (float32)."""
    ordered, order = logits.sort(descending=True)
    cumulative = ordered.softmax(-1).cumsum(-1)
    drop = (cumulative > top_p) | (torch.arange(len(ordered), device=logits.device) >= top_k)
    drop[0] = False  # always keep the most likely one
    logits = logits.masked_fill(drop.scatter(0, order, drop), float("-inf"))
    probs = (logits / max(temperature, 1e-5)).softmax(-1)
    race = -torch.rand(probs.shape, device=probs.device, generator=generator).log()
    return int((probs / race).argmax())


def generate(
    model: DualAR,
    prompt: Tensor,
    end_token: int,
    sampling: Sampling = DEFAULT_SAMPLING,
    max_frames: int = 1024,
    on_frame: Callable[[], None] | None = None,
) -> Tensor:
    """Frames after ``prompt`` [codebooks + 1, T] until the model ends its turn:
    codes [codebooks, n]."""
    c = model.config
    device = model.embeddings.weight.device
    prompt = prompt.to(device)
    start = prompt.shape[1]
    model.allocate(start + max_frames + 1)
    # The semantic tokens and the turn's end are the only tokens a frame may start with.
    allowed = torch.cat(
        [torch.arange(c.semantic_begin, c.semantic_end + 1, device=device), torch.tensor([end_token], device=device)]
    )
    recent: deque[int] = deque(maxlen=RAS_WINDOW)
    frames: list[Tensor] = []
    s = sampling
    with torch.inference_mode():
        logits, hidden = model.slow(prompt, 0)
        for step in range(max_frames):
            choices = logits.float()[allowed]
            token = int(allowed[draw(choices, s.temperature, s.top_p, s.top_k)])
            if step and token != end_token and token in recent:
                token = int(allowed[draw(choices, RAS_TEMPERATURE, RAS_TOP_P, s.top_k)])
            if token == end_token:
                break
            codes = [token - c.semantic_begin]
            model.fast(hidden, 0)
            for position in range(1, c.codebooks):
                x = model.fast_embeddings.weight[codes[-1]]
                codes.append(draw(model.fast(x, position).float(), s.temperature, s.top_p, s.top_k))
            frame = torch.tensor([token, *codes], device=device)
            frames.append(frame[1:])
            if step:
                recent.append(token)
            if on_frame:
                on_frame()
            logits, hidden = model.slow(frame[:, None], start + step)
    if not frames:
        return torch.zeros(c.codebooks, 0, dtype=torch.long)
    return torch.stack(frames, 1).cpu()
