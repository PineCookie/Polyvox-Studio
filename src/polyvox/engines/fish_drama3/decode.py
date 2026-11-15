"""Drama 3 decoding with one CUDA graph per audio frame.

The prompt goes through transformers' modules once (model.prefill); its states are copied into
static buffers, and each frame (main token with the repetition guard, the tower's nine codes,
the backbone step on the new frame) is recorded once and replayed. The backbone step is our own,
on the same weights: Gated DeltaNet layers advance their convolution and recurrent states, the
gated full-attention layers attend over the first ``window`` positions of a static cache. A
token's experts are read from their GPU slots or from CPU memory inside the kernel
(Experts.routed), so a step decides nothing on the CPU; between frames ``plan_experts`` moves
the most used experts to the GPU now and then. Samples like Drama3.generate_reference given the
same random numbers (they are drawn in another order).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import CAPACITY, CODES, DEFAULT_SAMPLING, RECENT, REDRAW, Drama3, Sampling

WINDOWS = (512, 1024, 2048, CAPACITY)
REPLAN = 128  # frames between refreshes of the experts held on the GPU


def pick(logits: Tensor, temperature: Tensor, top_p: Tensor, top_k: Tensor) -> Tensor:
    """model.sample with its settings as tensors (graph-safe): an index [1] long."""
    x = logits / temperature.clamp(min=1e-5)
    ordered, order = x.sort(descending=True)
    cumulative = ordered.softmax(-1).cumsum(-1)
    rank = torch.arange(len(ordered), device=logits.device)
    drop = ((cumulative > top_p) | (rank >= top_k)) & (rank > 0)  # the most likely always stays
    p = ordered.masked_fill(drop, float("-inf")).softmax(-1)
    choice = (p / torch.empty_like(p).exponential_(1)).argmax()
    return order.index_select(0, choice.view(1))  # index_select: a 0-d index would be a sync


def rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """transformers' rotate-half rotary on the first cos.shape[-1] dims of x [heads, d]."""
    rd = cos.shape[-1]
    r, rest = x[..., :rd], x[..., rd:]
    half = rd // 2
    rotated = torch.cat((-r[..., half:], r[..., :half]), -1)
    return torch.cat((r * cos + rotated * sin, rest), -1)


def l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


class Backbone:
    """The backbone's step for one token, with its states in static buffers."""

    def __init__(self, m: Drama3, capacity: int = CAPACITY):
        c = m.text_config
        self.m, self.layers, self.kinds = m, m.backbone.layers, c.layer_types
        dev = m.device
        dtype = m.backbone.embed_tokens.weight.dtype
        self.heads, self.kv_heads, self.head_dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.nk, self.nv = c.linear_num_key_heads, c.linear_num_value_heads
        self.dk, self.dv = c.linear_key_head_dim, c.linear_value_head_dim
        conv_dim = 2 * self.nk * self.dk + self.nv * self.dv
        self.k, self.v, self.conv, self.rec = {}, {}, {}, {}
        for i, kind in enumerate(self.kinds):
            if kind == "full_attention":
                self.k[i] = torch.zeros(self.kv_heads, capacity, self.head_dim, dtype=dtype, device=dev)
                self.v[i] = torch.zeros_like(self.k[i])
            else:
                self.conv[i] = torch.zeros(conv_dim, c.linear_conv_kernel_dim, dtype=dtype, device=dev)
                self.rec[i] = torch.zeros(self.nv, self.dk, self.dv, dtype=torch.float32, device=dev)
        inv_freq = m.backbone.rotary_emb.inv_freq.to(dev, torch.float32)  # (kept on the CPU by transformers)
        freqs = torch.arange(capacity, dtype=torch.float32, device=dev)[:, None] * inv_freq
        angles = torch.cat((freqs, freqs), -1)
        self.cos, self.sin = angles.cos().to(dtype), angles.sin().to(dtype)  # [capacity, rotary_dim]
        self.span = torch.arange(capacity, device=dev)
        self.scale = self.head_dim**-0.5

    def states(self) -> list[Tensor]:
        return [*self.k.values(), *self.v.values(), *self.conv.values(), *self.rec.values()]

    def load(self, cache, n: int) -> None:
        """The prompt's states from transformers' cache (``n`` positions)."""
        for i, layer in enumerate(cache.layers):
            if i in self.k:
                self.k[i][:, :n].copy_(layer.keys[0])
                self.v[i][:, :n].copy_(layer.values[0])
            else:
                self.conv[i].copy_(layer.conv_states[0][0])
                self.rec[i].copy_(layer.recurrent_states[0][0])

    def step(self, x: Tensor, pos: Tensor, window: int) -> Tensor:
        """x [1, dim] at ``pos`` [1] -> the final-norm hidden state [1, dim]."""
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            mix = (
                self._attention(i, layer.self_attn, h, pos, window)
                if i in self.k
                else self._delta(i, layer.linear_attn, h)
            )
            x = x + mix
            x = x + self._moe(layer.mlp, layer.post_attention_layernorm(x))
        return self.m.backbone.norm(x)

    def _delta(self, i: int, a, h: Tensor) -> Tensor:
        """Gated DeltaNet, one token (transformers' cached-decode path)."""
        nk, nv, dk, dv = self.nk, self.nv, self.dk, self.dv
        qkv, z = a.in_proj_qkv(h), a.in_proj_z(h)
        b, g_in = a.in_proj_b(h), a.in_proj_a(h)
        state = self.conv[i]  # the last kernel-size inputs
        state.copy_(torch.cat((state[:, 1:], qkv.T.to(state.dtype)), -1))
        mixed = F.silu((state.float() * a.conv1d.weight[:, 0].float()).sum(-1).to(h.dtype))
        q, k, v = mixed.split([nk * dk, nk * dk, nv * dv], -1)
        rep = nv // nk
        q = l2norm(q.view(nk, dk).repeat_interleave(rep, 0).float()) / math.sqrt(dk)
        k = l2norm(k.view(nk, dk).repeat_interleave(rep, 0).float())
        beta = b.sigmoid().float()[0]
        decay = (-a.A_log.float().exp() * F.softplus(g_in.float() + a.dt_bias))[0].exp()
        s = self.rec[i]
        s.mul_(decay[:, None, None])
        delta = (v.view(nv, dv).float() - (s * k[:, :, None]).sum(-2)) * beta[:, None]
        s.add_(k[:, :, None] * delta[:, None, :])
        out = (s * q[:, :, None]).sum(-2).to(h.dtype)  # [nv, dv]
        return a.out_proj(a.norm(out, z.view(nv, dv)).reshape(1, nv * dv))

    def _attention(self, i: int, a, h: Tensor, pos: Tensor, window: int) -> Tensor:
        """Gated full attention, one token: its key / value written at ``pos``."""
        H, KV, d = self.heads, self.kv_heads, self.head_dim
        q, gate = a.q_proj(h).view(H, 2 * d).chunk(2, -1)
        cos, sin = self.cos.index_select(0, pos), self.sin.index_select(0, pos)  # [1, rotary_dim]
        q = rope(a.q_norm(q), cos, sin)
        k = rope(a.k_norm(a.k_proj(h).view(KV, d)), cos, sin)
        self.k[i].index_copy_(1, pos, k[:, None])
        self.v[i].index_copy_(1, pos, a.v_proj(h).view(KV, 1, d))
        # The same attention call as the model's prompt pass (SDPA), so steps round as it does.
        y = F.scaled_dot_product_attention(
            q[None, :, None], self.k[i][None, :, :window], self.v[i][None, :, :window],
            attn_mask=(self.span[:window] <= pos)[None], scale=self.scale, enable_gqa=True,
        ).reshape(1, H * d)  # fmt: skip
        return a.o_proj(y * torch.sigmoid(gate.reshape(1, H * d)))

    def _moe(self, mlp, h: Tensor) -> Tensor:
        probs = F.linear(h, mlp.gate.weight).softmax(-1, dtype=torch.float32)
        top, experts = probs.topk(mlp.gate.top_k, -1)
        top = (top / top.sum(-1, keepdim=True)).to(h.dtype)
        routed = mlp.experts.routed(h, experts[0], top[0])
        return routed + torch.sigmoid(mlp.shared_expert_gate(h)) * mlp.shared_expert(h)


class Decoder:
    def __init__(self, m: Drama3, graphs: bool = True):
        self.m, self.graphs, self.device = m, graphs, m.device
        dev = self.device
        self.backbone = Backbone(m)
        dim = m.text_config.hidden_size
        dtype = m.backbone.embed_tokens.weight.dtype

        def scalar(value, dtype=torch.long):
            return torch.tensor(value, dtype=dtype, device=dev)

        # The state a step reads and advances.
        self.h = torch.zeros(1, dim, dtype=dtype, device=dev)  # hidden state at pos
        self.frame = torch.zeros(1, CODES + 1, dtype=torch.long, device=dev)  # the row at pos
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.recent = torch.full((RECENT,), -1, dtype=torch.long, device=dev)
        self.temperature, self.top_p, self.top_k = scalar(1.0, torch.float32), scalar(0.9, torch.float32), scalar(30)
        self.redraw = (
            scalar(REDRAW.temperature, torch.float32),
            scalar(REDRAW.top_p, torch.float32),
            scalar(REDRAW.top_k),
        )
        self.no_codes = torch.zeros(CODES - 1, dtype=torch.long, device=dev)
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    # ------------------------------------------------------------ one frame
    def _step(self, window: int) -> None:
        m = self.m
        logits = F.linear(self.h, m.head)[0].float()
        first = m.candidates.index_select(0, pick(logits, self.temperature, self.top_p, self.top_k))[0]
        again = m.candidates.index_select(0, pick(logits, *self.redraw))[0]
        main = torch.where(m.is_semantic(first) & (self.recent == first).any(), again, first)
        self.recent.copy_(torch.cat([self.recent[1:], main.view(1)]))
        code0 = (main - m.semantic_begin).clamp(0, 4095)
        label = torch.cat([main.view(1), code0.view(1), self.no_codes]).view(1, CODES + 1)

        def draw(lg):
            return pick(lg[0].float(), self.temperature, self.top_p, self.top_k)

        codes = m.tower(self.h, self.pos, self.frame, label, window, draw)
        frame = torch.cat([main.view(1), codes[0]]).view(1, CODES + 1)
        self.frame.copy_(frame)
        self.pos.add_(1)
        self.h.copy_(self.backbone.step(m.backbone_input(frame), self.pos, window))

    # ------------------------------------------------------------ graphs
    @property
    def warm(self) -> bool:
        return not self.graphs or bool(self._graphs)

    def _run(self, window: int) -> None:
        if not self.graphs:
            self._step(window)
            return
        if window not in self._graphs:
            self._capture(window)
        self._graphs[window].replay()

    def _state(self) -> list[Tensor]:
        m = self.m
        tower = [t for layer in (*m.shared, *m.residual) for t in (layer.k, layer.v)]
        counts = [layer.mlp.experts.counts for layer in m.backbone.layers]
        return [self.h, self.frame, self.pos, self.recent, *self.backbone.states(), *tower, *counts]

    def _capture(self, window: int) -> None:
        """Record one step; the state and the random generator are put back afterwards, so a
        seed samples the same whether or not a call captured."""
        state = self._state()
        saved = [t.clone() for t in state]
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
        torch.cuda.set_rng_state(rng, self.device)
        self._graphs[window] = graph

    # ------------------------------------------------------------ decoding
    @torch.inference_mode()
    def generate(
        self,
        rows: Tensor,
        sampling: Sampling = DEFAULT_SAMPLING,
        max_frames: int = 1024,
        on_frame: Callable[[], None] | None = None,
        head: int = 0,
    ) -> Tensor:
        """As Drama3.generate_reference: codes [10, n] after the prompt ``rows``."""
        m = self.m
        rows = rows.to(self.device)
        n = rows.shape[0]
        if n + max_frames + 1 > CAPACITY:
            raise ValueError(f"The prompt is too long for Drama 3 ({n} positions).")
        self.temperature.fill_(sampling.temperature)
        self.top_p.fill_(sampling.top_p)
        self.top_k.fill_(sampling.top_k)
        h, cache = m.prefill(rows, head)
        self.backbone.load(cache, n)
        del cache
        m.plan_experts()  # the experts the prompt used most
        self.h.copy_(h)
        self.frame.copy_(rows[-1:])
        self.pos.fill_(n - 1)
        self.recent.fill_(-1)
        frames: list[Tensor] = []
        for step in range(max_frames):
            window = next(w for w in WINDOWS if w > n + step)
            self._run(window)
            frame = self.frame[0].clone()
            if int(frame[0]) == m.eos:
                break
            frames.append(frame[1:])
            if on_frame:
                on_frame()
            if (step + 1) % REPLAN == 0:
                m.plan_experts()
        if not frames:
            return torch.zeros(CODES, 0, dtype=torch.long)
        return torch.stack(frames, 1).cpu()
