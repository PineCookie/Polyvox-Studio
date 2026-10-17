"""Training sequences and losses for S2-Pro LoRAs (the fast transformer; see lora.py).

A training sequence is the inference prompt (prompt.build: system turn with or
without a reference, user text, assistant voice turn), the target clip's frames,
then <|im_end|>. The slow transformer runs over whole sequences (causal,
right-padded batches; without gradients unless it trains too); the fast transformer then runs on
every target frame at once: the slow hidden state before the frame, then the
frame's codes 0..8 embedded, predicting codes 1..9 (code 0 is the slow
transformer's). Both match the decoder within bfloat16 rounding.

Clips are encoded with a faint noise floor (NOISE_FLOOR_DB): datasets of
separated, gated vocals hold digital silence the base model rarely heard, and a
LoRA trained on it paused for seconds mid-sentence.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from . import prompt
from .model import DualAR, rope_table, rotate

NOISE_FLOOR_DB = -72.0  # dBFS of white noise added to training clips (and to trained voices' references)
REFERENCE_SHARE = 0.5  # of training sequences that clone from other clips of the voice
MAX_REFERENCE_CLIPS = 3
# Training only the fast transformer: references cached per clip (slow_states), and the
# padded positions the slow transformer runs at a time to make the cache.
CACHED_REFERENCES = 2
CACHE_TOKENS = 3072
FRAME_ROUND = 64  # training batches' frame counts are rounded up to a multiple of this
_ENDS = "。！？，.!?,"


def with_noise_floor(audio: Tensor, seed: int = 0, db: float = NOISE_FLOOR_DB) -> Tensor:
    g = torch.Generator(device=audio.device).manual_seed(seed)
    return audio + torch.randn(audio.shape, device=audio.device, generator=g, dtype=audio.dtype) * 10 ** (db / 20)


class Sequences:
    """Token rows [codebooks + 1, T] of training sequences."""

    def __init__(self, tokenizer, model: DualAR, end_token: int):
        self.tokenizer, self.config, self.end_token = tokenizer, model.config, end_token

    def rows(self, text: str, codes: Tensor, reference: tuple[Tensor, str] | None) -> tuple[Tensor, int]:
        """(rows, where the target frames start)."""
        c = self.config
        head = prompt.build(self.tokenizer, c.codebooks, c.semantic_begin, text, reference)
        codes = codes.long()
        end = torch.zeros(c.codebooks + 1, 1, dtype=torch.long)
        end[0, 0] = self.end_token
        frames = torch.cat([codes[:1] + c.semantic_begin, codes])
        return torch.cat([head, frames, end], 1), head.shape[1]


def joined_reference(clips: list[dict], chosen: list[int]) -> tuple[Tensor, str]:
    """Clips' codes and transcripts joined into one reference."""
    codes = torch.cat([clips[j]["codes"].long() for j in chosen], 1)
    text = "".join(t if t[-1:] in _ENDS else t + "。" for t in (clips[j]["text"].strip() for j in chosen))
    return codes, text


def pick_reference(clips: list[dict], exclude: int, rng: random.Random) -> tuple[Tensor, str]:
    """1..MAX_REFERENCE_CLIPS other clips of the voice: codes and transcripts joined."""
    others = [j for j in range(len(clips)) if j != exclude]
    return joined_reference(clips, rng.sample(others, min(len(others), rng.randint(1, MAX_REFERENCE_CLIPS))))


def reference_picks(keys: list[str], references: int = CACHED_REFERENCES) -> list[list[list[int]]]:
    """Per clip, ``references`` picks of 1..MAX_REFERENCE_CLIPS other clips (indices), the
    same for the same clips in any run: seeded by the clip's key, drawn from the clips
    in key order."""
    order = sorted(range(len(keys)), key=lambda j: keys[j])
    picks = []
    for i, key in enumerate(keys):
        rng = random.Random(int(key[:12], 16))
        others = [j for j in order if j != i]
        picks.append(
            [rng.sample(others, min(len(others), rng.randint(1, MAX_REFERENCE_CLIPS))) for _ in range(references)]
        )
    return picks


def slow_states(
    trainer: Trainer, seq: Sequences, clips: list[dict], picks: list[list[list[int]]], todo: list[int],
    tokens: int = CACHE_TOKENS, stop=None,
) -> dict[int, list[Tensor]] | None:  # fmt: skip
    """For training only the fast transformer, whose input the frozen slow one gives the
    same way every epoch: for each clip in ``todo``, the slow states before its frames
    (CPU, [frames, dim]): [0] without a reference, then one per pick (reference_picks)
    rather than a new pick each time the clip comes up. Sequences of similar length
    run together, ``tokens`` padded positions at a time. None: stopped."""
    jobs = []
    for i in todo:
        c = clips[i]
        jobs.append((i, 0, *seq.rows(c["text"], c["codes"], None)))
        for v, chosen in enumerate(picks[i], 1):
            jobs.append((i, v, *seq.rows(c["text"], c["codes"], joined_reference(clips, chosen))))
    jobs.sort(key=lambda j: j[2].shape[1])  # similar lengths together: little padding
    out = {i: [None] * (len(picks[i]) + 1) for i in todo}
    device = trainer.model.embeddings.weight.device
    k = 0
    while k < len(jobs):
        if stop is not None and stop.is_set():
            return None
        chunk = [jobs[k]]  # ascending lengths: the last one added is the longest
        while k + len(chunk) < len(jobs) and (len(chunk) + 1) * jobs[k + len(chunk)][2].shape[1] <= tokens:
            chunk.append(jobs[k + len(chunk)])
        frames = trainer.slow_frames(*collate([(rows, start) for _, _, rows, start in chunk], device))
        for (i, v, _, _), f in zip(chunk, frames, strict=True):
            out[i][v] = f.cpu()
        k += len(chunk)
    return out


def collate(items: list[tuple[Tensor, int]], device) -> tuple[Tensor, list[tuple[int, int]]]:
    """[B, codebooks + 1, L] right-padded; per item (start, frames)."""
    length = max(r.shape[1] for r, _ in items)
    out = torch.zeros(len(items), items[0][0].shape[0], length, dtype=torch.long)
    spans = []
    for i, (r, start) in enumerate(items):
        out[i, :, : r.shape[1]] = r
        spans.append((start, r.shape[1] - start - 1))  # the frames between the prompt and the end token
    return out.to(device, non_blocking=True), spans


def embed(m: DualAR, tokens: Tensor) -> Tensor:
    """model.DualAR.embed for a batch: [B, codebooks + 1, L] -> [B, L, dim]."""
    c = m.config
    x = m.embeddings(tokens[:, 0])
    audio = (tokens[:, 0] >= c.semantic_begin) & (tokens[:, 0] <= c.semantic_end)
    offsets = (torch.arange(c.codebooks, device=tokens.device) * c.codebook_size)[None, :, None]
    codes = m.codebook_embeddings(tokens[:, 1:] + offsets).sum(1)
    return torch.where(audio[..., None], (x + codes) / math.sqrt(c.codebooks + 1), x)


def slow_layer(layer, x: Tensor, table: Tensor) -> Tensor:
    a, s = layer.attention, layer.attention.s
    b, t, _ = x.shape
    h = layer.attention_norm(x)
    q, k, v = a.wqkv(h).split([s.heads * s.head_dim, s.kv_heads * s.head_dim, s.kv_heads * s.head_dim], -1)
    q = a.q_norm(q.view(b, t, s.heads, s.head_dim))
    k = a.k_norm(k.view(b, t, s.kv_heads, s.head_dim))
    q, k = rotate(q, table).transpose(1, 2), rotate(k, table).transpose(1, 2)
    v = v.view(b, t, s.kv_heads, s.head_dim).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    x = x + a.wo(y.transpose(1, 2).reshape(b, t, -1))
    return x + layer.feed_forward(layer.ffn_norm(x))


def slow_forward(m: DualAR, tokens: Tensor, table: Tensor, recompute: bool = False) -> Tensor:
    """Normalized hidden states [B, L, dim] (right padding never reaches real positions);
    ``recompute``: with gradients, each layer's activations recomputed in the backward pass."""
    x = embed(m, tokens)
    table = table[: tokens.shape[2]]
    for layer in m.layers:
        if recompute and torch.is_grad_enabled():
            x = checkpoint(slow_layer, layer, x, table, use_reentrant=False)
        else:
            x = slow_layer(layer, x, table)
    return m.norm(x)


def fast_layer(layer, x: Tensor, table: Tensor) -> Tensor:
    a, s = layer.attention, layer.attention.s
    n, t, _ = x.shape
    h = layer.attention_norm(x)
    q, k, v = a.wqkv(h).split([s.heads * s.head_dim, s.kv_heads * s.head_dim, s.kv_heads * s.head_dim], -1)
    q = rotate(q.view(n, t, s.heads, s.head_dim), table).transpose(1, 2)
    k = rotate(k.view(n, t, s.kv_heads, s.head_dim), table).transpose(1, 2)
    v = v.view(n, t, s.kv_heads, s.head_dim).transpose(1, 2)
    groups = s.heads // s.kv_heads
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
    scores = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(s.head_dim))  # plain bfloat16, as trained
    y = torch.softmax(scores.masked_fill(~mask, float("-inf")), -1) @ v
    x = x + a.wo(y.transpose(1, 2).reshape(n, t, -1))
    return x + layer.feed_forward(layer.ffn_norm(x))


def recomputed_fast_layer(layer, x: Tensor, table: Tensor) -> Tensor:
    """fast_layer with its activations recomputed in the backward pass (every frame of
    a batch at once is a lot of positions)."""
    return checkpoint(fast_layer, layer, x, table, use_reentrant=False)


def fast_forward(m: DualAR, x: Tensor, table: Tensor, layer_fn=fast_layer) -> Tensor:
    """x [N, codebooks, dim] -> code logits [N, codebooks, codebook_size]; ``layer_fn``
    runs each layer while training with gradients (plain fast_layer otherwise)."""
    run = layer_fn if torch.is_grad_enabled() else fast_layer
    for layer in m.fast_layers:
        x = run(layer, x, table)
    return m.fast_output(m.fast_norm(x))


class Trainer:
    """``slow``: the slow transformer trains too (a LoRA on it, lora.py "slow_attn"):
    it runs with gradients, and its choice of each frame's semantic token and of
    the end is scored as well, among the tokens the decoder chooses from.

    ``fused``: the fast layers' training forward (with its recomputation) as one
    torch.compile'd function for every layer and batch size, which fuses the small
    operations around the matrix products. If compiling fails before a fused call
    has worked, training goes on unfused."""

    def __init__(self, model: DualAR, end_token: int, slow: bool = False, recompute: bool = True, fused: bool = False):
        c = model.config
        dev = model.embeddings.weight.device
        self.model, self.slow, self.recompute = model, slow, recompute
        self.slow_table = rope_table(4096, c.slow.head_dim, c.slow.rope_base, dev)
        self.fast_table = rope_table(c.codebooks, c.fast.head_dim, c.fast.rope_base, dev)
        self.allowed = torch.cat(
            [torch.arange(c.semantic_begin, c.semantic_end + 1, device=dev), torch.tensor([end_token], device=dev)]
        )
        self._layer = recomputed_fast_layer if recompute else fast_layer
        self._compiled = None
        self._fused_worked = False
        if fused:
            from ..base import fuse_settings

            fuse_settings()
            self._compiled = torch.compile(self._layer, dynamic=True)

    def _fast_layer(self, layer, x: Tensor, table: Tensor) -> Tensor:
        if self._compiled is not None:
            try:
                out = self._compiled(layer, x, table)
                self._fused_worked = True
                return out
            except Exception as e:
                if self._fused_worked or isinstance(e, torch.OutOfMemoryError):
                    raise
                import logging

                logging.getLogger(__name__).warning("Fusing the fast layers failed (%r); running them unfused.", e)
                self._compiled = None
        return self._layer(layer, x, table)

    def slow_frames(self, tokens: Tensor, spans: list[tuple[int, int]]) -> list[Tensor]:
        """The slow transformer's normalized hidden state before each target frame, per
        sequence ([frames, dim]; without gradients): what the fast transformer starts from."""
        with torch.no_grad():
            hidden = slow_forward(self.model, tokens, self.slow_table)
        return [hidden[i, start - 1 : start + n - 1] for i, (start, n) in enumerate(spans)]

    def code_loss(self, hidden: Tensor, codes: Tensor) -> tuple[Tensor, int]:
        """(summed cross-entropy of codes 1..9, how many) of frames: ``hidden`` [N, dim] the
        slow state before each, ``codes`` [N, codebooks] their codes."""
        m, c = self.model, self.model.config
        n = len(codes)
        if torch.is_grad_enabled():
            # Frame counts rounded up (padding frames left out of the loss): fewer distinct
            # tensor sizes, so the memory allocator reuses its blocks instead of piling them up.
            pad = -n % FRAME_ROUND
            hidden, codes = F.pad(hidden, (0, 0, 0, pad)), F.pad(codes, (0, 0, 0, pad))
        x = torch.cat([hidden[:, None], m.fast_embeddings(codes[:, :-1])], 1)
        logits = fast_forward(m, x, self.fast_table, self._fast_layer)[:, 1:].float()
        targets = codes[:, 1:].clone()
        targets[n:] = -100  # ignored by cross_entropy
        loss = F.cross_entropy(logits.reshape(-1, c.codebook_size), targets.reshape(-1), reduction="sum")
        return loss, n * (c.codebooks - 1)

    def losses(self, tokens: Tensor, spans: list[tuple[int, int]]) -> dict[str, tuple[Tensor, int]]:
        """{"codes": (summed cross-entropy of the target frames' codes 1..9, how many)}, and
        with ``slow`` "semantic": the same for each frame's semantic token and the end."""
        m, c = self.model, self.model.config
        if tokens.shape[2] > len(self.slow_table):
            raise ValueError(f"A training sequence is too long ({tokens.shape[2]} positions).")
        with torch.set_grad_enabled(self.slow and torch.is_grad_enabled()):
            hidden = slow_forward(m, tokens, self.slow_table, self.recompute)
        dev = tokens.device
        batch = torch.cat([torch.full((n,), i) for i, (_, n) in enumerate(spans)]).to(dev)
        frame = torch.cat([torch.arange(start, start + n) for start, n in spans]).to(dev)
        out = {"codes": self.code_loss(hidden[batch, frame - 1], tokens[batch, 1:, frame])}
        if self.slow:
            # Each frame's semantic token, then the end: predicted from the position before.
            end = len(self.allowed) - 1
            rows = torch.cat([torch.full((n + 1,), i) for i, (_, n) in enumerate(spans)]).to(dev)
            before = torch.cat([torch.arange(start - 1, start + n) for start, n in spans]).to(dev)
            wanted = torch.cat(
                [torch.cat([tokens[i, 0, start : start + n] - c.semantic_begin, torch.tensor([end], device=dev)])
                 for i, (start, n) in enumerate(spans)]
            )  # fmt: skip
            head = m.embeddings.weight.index_select(0, self.allowed)
            semantic = (hidden[rows, before] @ head.T).float()
            out["semantic"] = (F.cross_entropy(semantic, wanted, reduction="sum"), len(wanted))
        return out
