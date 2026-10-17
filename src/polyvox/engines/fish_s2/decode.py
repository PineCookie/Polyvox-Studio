"""S2-Pro decoding with one CUDA graph per audio frame.

Eager decoding launches ~1500 small kernels per frame (the slow transformer's 36
layers, then the fast transformer's 4 layers ten times) and the GPU mostly waits
on Python. Here one frame (slow step, semantic-token sampling with the
repetition guard, the ten fast steps and their sampling) is recorded once and
replayed: the state it reads and advances lives on the device. A step attends
over the first ``window`` cache positions (masked past the frame's own), so
short texts don't pay for the whole cache; decoding moves to the next window
when it fills one. Samples the same as generate.generate given the same random
numbers (only the order the random numbers are drawn in differs).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .generate import DEFAULT_SAMPLING, RAS_TOP_P, RAS_WINDOW, Sampling
from .model import DualAR, rotate

CAPACITY = 4096  # positions: a 20 s reference, the text, the previous part and 1024 new frames
WINDOWS = (512, 1024, 2048)


def pick(logits: Tensor, temperature: Tensor, top_p: Tensor, top_k: Tensor) -> Tensor:
    """generate.draw with its settings as tensors (graph-safe): an index [] long."""
    ordered, order = logits.sort(descending=True)
    cumulative = ordered.softmax(-1).cumsum(-1)
    rank = torch.arange(len(ordered), device=logits.device)
    drop = ((cumulative > top_p) | (rank >= top_k)) & (rank > 0)  # the most likely always stays
    logits = logits.masked_fill(drop.scatter(0, order, drop), float("-inf"))
    probs = (logits / temperature.clamp(min=1e-5)).softmax(-1)
    return (probs / -torch.rand_like(probs).log()).argmax()


class Decoder:
    def __init__(self, model: DualAR, end_token: int, graphs: bool = True):
        c = model.config
        self.model, self.config, self.end_token, self.graphs = model, c, end_token, graphs
        weight = model.embeddings.weight
        dev = self.device = weight.device
        model.allocate(CAPACITY)
        self.windows = (*(w for w in WINDOWS if w < CAPACITY), CAPACITY)
        # A frame starts with a semantic token or the turn's end: the logits of those only.
        self.allowed = torch.cat(
            [torch.arange(c.semantic_begin, c.semantic_end + 1, device=dev), torch.tensor([end_token], device=dev)]
        )
        self.head = weight[self.allowed].contiguous()  # [allowed, dim]
        self.offsets = torch.arange(c.codebooks, device=dev) * c.codebook_size
        self.positions = torch.arange(CAPACITY, device=dev)

        def scalar(value, dtype=torch.long):
            return torch.tensor(value, dtype=dtype, device=dev)

        # The state a step reads and advances.
        self.frame = torch.zeros(c.codebooks + 1, dtype=torch.long, device=dev)  # last frame: token + codes
        self.pos = scalar(0)  # where the last frame goes in the cache
        self.recent = torch.full((RAS_WINDOW,), -1, dtype=torch.long, device=dev)  # semantic tokens drawn lately
        self.temperature = scalar(0.8, torch.float32)
        self.top_p = scalar(0.8, torch.float32)
        self.top_k = scalar(30)
        self.ras_temperature = scalar(1.0, torch.float32)
        self.ras_top_p = scalar(RAS_TOP_P, torch.float32)
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    # ------------------------------------------------------------ one frame
    def _slow(self, window: int) -> Tensor:
        """The last frame at ``pos`` through the slow transformer: its normalized hidden state [dim]."""
        m, c = self.model, self.config
        tokens = self.frame
        x = (m.embeddings(tokens[:1]) + m.codebook_embeddings(tokens[1:] + self.offsets).sum(0)) / math.sqrt(
            c.codebooks + 1
        )
        x = x.view(1, 1, -1)
        index = self.pos.view(1)
        table = m.slow_table.index_select(0, index)
        mask = (self.positions[:window] <= self.pos).view(1, 1, 1, window)
        s = c.slow
        for layer in m.layers:
            a = layer.attention
            h = layer.attention_norm(x)
            q, k, v = a.wqkv(h).split([s.heads * s.head_dim, s.kv_heads * s.head_dim, s.kv_heads * s.head_dim], -1)
            q = a.q_norm(q.view(1, 1, s.heads, s.head_dim))
            k = a.k_norm(k.view(1, 1, s.kv_heads, s.head_dim))
            q, k = rotate(q, table).transpose(1, 2), rotate(k, table).transpose(1, 2)
            ck, cv = a.cache
            ck.index_copy_(2, index, k)
            cv.index_copy_(2, index, v.view(1, 1, s.kv_heads, s.head_dim).transpose(1, 2))
            y = F.scaled_dot_product_attention(q, ck[:, :, :window], cv[:, :, :window], attn_mask=mask, enable_gqa=True)
            x = x + a.wo(y.transpose(1, 2).reshape(1, 1, -1))
            x = x + layer.feed_forward(layer.ffn_norm(x))
        return m.norm(x)[0, 0]

    def _frame(self, hidden: Tensor, guard: bool) -> None:
        """Sample the next frame from the slow hidden state into ``frame``."""
        m, c = self.model, self.config

        def choose(logits, temperature, top_p):  # index_select: a 0-d index would be a sync (no capture)
            return self.allowed.index_select(0, pick(logits, temperature, top_p, self.top_k).view(1))[0]

        logits = (hidden @ self.head.T).float()
        token = choose(logits, self.temperature, self.top_p)
        if guard:  # a semantic token among the last few: drawn again, more freely
            again = choose(logits, self.ras_temperature, self.ras_top_p)
            token = torch.where((self.recent == token).any() & (token != self.end_token), again, token)
            self.recent.copy_(torch.cat([self.recent[1:], token.view(1)]))
        code = (token - c.semantic_begin).clamp(0, c.codebook_size - 1)
        codes = [code]
        m.fast(hidden, 0)
        for position in range(1, c.codebooks):
            x = m.fast_embeddings.weight.index_select(0, code.view(1))[0]
            code = pick(m.fast(x, position).float(), self.temperature, self.top_p, self.top_k)
            codes.append(code)
        self.frame.copy_(torch.stack([token, *codes]))

    def _step(self, window: int) -> None:
        self._frame(self._slow(window), guard=True)
        self.pos.add_(1)

    # ------------------------------------------------------------ graphs
    def _run(self, window: int) -> None:
        if not self.graphs:
            self._step(window)
            return
        if window not in self._graphs:
            self._capture(window)
        self._graphs[window].replay()

    def _capture(self, window: int) -> None:
        """Record one step; the state and the random generator are put back
        afterwards, so a seed samples the same whether or not a call captured."""
        state = [self.frame, self.pos, self.recent]
        saved = [t.clone() for t in state]
        caches = [layer.attention.cache for layer in self.model.layers]
        saved_caches = [(k.clone(), v.clone()) for k, v in caches]
        rng = torch.cuda.get_rng_state(self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._step(window)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step(window)
        for t, s in zip(state, saved, strict=True):
            t.copy_(s)
        for (k, v), (sk, sv) in zip(caches, saved_caches, strict=True):
            k.copy_(sk)
            v.copy_(sv)
        torch.cuda.set_rng_state(rng, self.device)
        self._graphs[window] = graph

    # ------------------------------------------------------------ decoding
    @torch.inference_mode()
    def generate(
        self,
        prompt: Tensor,
        sampling: Sampling = DEFAULT_SAMPLING,
        max_frames: int = 1024,
        on_frame: Callable[[], None] | None = None,
    ) -> Tensor:
        """As generate.generate: codes [codebooks, n] after ``prompt``."""
        start = prompt.shape[1]
        if start + max_frames + 1 > CAPACITY:
            raise ValueError(f"The prompt is too long for S2-Pro ({start} positions).")
        self.temperature.fill_(sampling.temperature)
        self.top_p.fill_(sampling.top_p)
        self.top_k.fill_(sampling.top_k)
        self.recent.fill_(-1)
        _, hidden = self.model.slow(prompt.to(self.device), 0)
        self._frame(hidden, guard=False)  # the first frame: no repetition yet
        self.pos.fill_(start)
        frames: list[Tensor] = []
        for step in range(max_frames):
            if step:
                window = next(w for w in self.windows if w > start + step - 1)
                self._run(window)
            frame = self.frame.clone()
            if int(frame[0]) == self.end_token:
                break
            frames.append(frame[1:])
            if on_frame:
                on_frame()
        if not frames:
            return torch.zeros(self.config.codebooks, 0, dtype=torch.long)
        return torch.stack(frames, 1).cpu()
