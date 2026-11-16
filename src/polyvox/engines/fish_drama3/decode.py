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

A layer is its linear layers (module calls) and the small steps between them (Steps: norms with
the residual adds, the Gated DeltaNet update, the attention heads, routing, the tower's steps),
which ``fused`` compiles with torch.compile: one compiled function per step serves every layer.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import CAPACITY, CODES, DEFAULT_SAMPLING, RECENT, REDRAW, Drama3, Sampling, linear

WINDOWS = (512, 1024, 2048, CAPACITY)
REPLAN = 128  # frames between refreshes of the experts held on the GPU
EPS = 1e-6  # every norm's


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


def zc_norm(x: Tensor, w: Tensor) -> Tensor:
    """The zero-centred RMSNorm of the backbone and the tower (model.gemma_norm)."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w.float())).to(x.dtype)


@dataclass(frozen=True)
class Steps:
    """The steps between a step's linear layers (one token)."""

    norm: Callable  # (x, w) -> x normalized
    add_norm: Callable  # (x, y, w) -> x + y and it normalized
    delta: (
        Callable  # Gated DeltaNet: (qkv, z, b, a, conv state, conv w, A_log, dt_bias, state, norm w) -> [1, value dim]
    )
    heads: Callable  # attention: (q and gate, k, v, q_norm, k_norm, cos, sin) -> q, gate, k, v
    gate: Callable  # (y, gate) -> y * sigmoid(gate)
    route: Callable  # router logits -> (weights, experts) [1, top_k]
    swiglu: Callable  # (a, b) -> silu(a) * b
    combine: Callable  # (expert outputs [k, dim] fp32, weights [k], shared [1, dim], shared gate [1, 1]) -> [1, dim]
    tower_in: Callable  # (emb, hidden, enorm, hnorm) -> both normalized, side by side
    tower_heads: Callable  # (qkv, q_norm, k_norm, cos, sin) -> q, gate, k, v
    tower_out: Callable  # (ff, residual, w) -> (ff + residual) normalized

    @staticmethod
    def of(m: Drama3, fused: bool) -> Steps:
        c, ts = m.text_config, m.shared[0].shape
        nk, nv, dk, dv = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        )
        H, KV, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        top_k = c.num_experts_per_tok

        def norm(x, w):
            return zc_norm(x, w)

        def add_norm(x, y, w):
            x = x + y
            return x, zc_norm(x, w)

        def delta(qkv, z, b, g_in, conv_state, conv_w, a_log, dt_bias, rec, norm_w):
            dtype = qkv.dtype
            new = torch.cat((conv_state[:, 1:], qkv.T.to(conv_state.dtype)), -1)  # the last kernel-size inputs
            conv_state.copy_(new)
            mixed = F.silu((new.float() * conv_w.float()).sum(-1).to(dtype))
            q, k, v = mixed.split([nk * dk, nk * dk, nv * dv], -1)
            rep = nv // nk
            q = l2norm(q.view(nk, dk).repeat_interleave(rep, 0).float()) / math.sqrt(dk)
            k = l2norm(k.view(nk, dk).repeat_interleave(rep, 0).float())
            beta = b.sigmoid().float()[0]
            decay = (-a_log.float().exp() * F.softplus(g_in.float() + dt_bias))[0].exp()
            s = rec * decay[:, None, None]
            s = s + k[:, :, None] * ((v.view(nv, dv).float() - (s * k[:, :, None]).sum(-2)) * beta[:, None])[:, None, :]
            rec.copy_(s)
            out = (s * q[:, :, None]).sum(-2).to(dtype)  # [nv, dv]
            # the gated RMSNorm (transformers' Qwen3_5MoeRMSNormGated)
            x = out.float()
            x = norm_w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)).to(dtype)
            return (x * F.silu(z.view(nv, dv).float())).to(dtype).reshape(1, nv * dv)

        def heads(qg, k, v, q_norm, k_norm, cos, sin):
            q, gate = qg.view(H, 2 * d).chunk(2, -1)
            return (rope(zc_norm(q, q_norm), cos, sin), gate.reshape(1, H * d),
                    rope(zc_norm(k.view(KV, d), k_norm), cos, sin), v.view(KV, d))  # fmt: skip

        def gate(y, g):
            return y * torch.sigmoid(g)

        def route(logits):
            probs = logits.softmax(-1, dtype=torch.float32)
            top, experts = probs.topk(top_k, -1)
            return (top / top.sum(-1, keepdim=True)).to(logits.dtype), experts

        def swiglu(a, b):
            return F.silu(a) * b

        def combine(down, weights, shared, shared_gate):
            routed = (down * weights.float()[:, None]).sum(0, keepdim=True).to(shared.dtype)
            return routed + torch.sigmoid(shared_gate) * shared

        def tower_in(emb, hidden, enorm, hnorm):
            return torch.cat((zc_norm(emb, enorm), zc_norm(hidden, hnorm)), -1)

        def tower_heads(qkv, q_norm, k_norm, cos, sin):
            qw, kw = ts.heads * ts.head_dim, ts.kv_heads * ts.head_dim
            q, g, k, v = qkv.split([qw, qw, kw, kw], -1)

            def rotate(x):  # interleaved pairs on the first rotary_dim dims (TowerLayer.rotate)
                r = x[..., : ts.rotary_dim].float()
                even, odd = r[..., 0::2], r[..., 1::2]
                rot = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
                return torch.cat((rot.to(x.dtype), x[..., ts.rotary_dim :]), -1)

            q = rotate(zc_norm(q.view(ts.heads, ts.head_dim), q_norm))
            k = rotate(zc_norm(k.view(ts.kv_heads, ts.head_dim), k_norm))
            return q, g, k, v.view(ts.kv_heads, ts.head_dim)

        def tower_out(ff, residual, w):
            return zc_norm(ff + residual, w)

        steps = (norm, add_norm, delta, heads, gate, route, swiglu, combine, tower_in, tower_heads, tower_out)
        if fused:  # rounding to bf16 where eager rounds (Inductor skips it inside a kernel by default)
            options = {"emulate_precision_casts": True}
            steps = tuple(torch.compile(f, dynamic=False, fullgraph=True, options=options) for f in steps)
        return Steps(*steps)


class Backbone:
    """The backbone's step for one token, with its states in static buffers."""

    def __init__(self, m: Drama3, steps: Steps, capacity: int = CAPACITY):
        c = m.text_config
        self.m, self.s, self.layers, self.kinds = m, steps, m.backbone.layers, c.layer_types
        dev = m.device
        dtype = m.backbone.embed_tokens.weight.dtype
        self.heads, self.kv_heads, self.head_dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        nk, nv, dk, dv = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        )
        self.k, self.v, self.conv, self.rec = {}, {}, {}, {}
        for i, kind in enumerate(self.kinds):
            if kind == "full_attention":
                self.k[i] = torch.zeros(self.kv_heads, capacity, self.head_dim, dtype=dtype, device=dev)
                self.v[i] = torch.zeros_like(self.k[i])
            else:
                self.conv[i] = torch.zeros(2 * nk * dk + nv * dv, c.linear_conv_kernel_dim, dtype=dtype, device=dev)
                self.rec[i] = torch.zeros(nv, dk, dv, dtype=torch.float32, device=dev)
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
        s, layers = self.s, self.layers
        h = s.norm(x, layers[0].input_layernorm.weight)
        for i, layer in enumerate(layers):
            if i in self.k:
                mix = self._attention(i, layer.self_attn, h, pos, window)
            else:
                a = layer.linear_attn
                mix = a.out_proj(
                    s.delta(a.in_proj_qkv(h), a.in_proj_z(h), a.in_proj_b(h), a.in_proj_a(h), self.conv[i],
                            a.conv1d.weight[:, 0], a.A_log, a.dt_bias, self.rec[i], a.norm.weight)
                )  # fmt: skip
            x, h = s.add_norm(x, mix, layer.post_attention_layernorm.weight)
            after = layers[i + 1].input_layernorm.weight if i + 1 < len(layers) else self.m.backbone.norm.weight
            x, h = s.add_norm(x, self._moe(layer.mlp, h), after)
        return h

    def _attention(self, i: int, a, h: Tensor, pos: Tensor, window: int) -> Tensor:
        """Gated full attention, one token: its key / value written at ``pos``."""
        cos, sin = self.cos.index_select(0, pos), self.sin.index_select(0, pos)  # [1, rotary_dim]
        q, gate, k, v = self.s.heads(a.q_proj(h), a.k_proj(h), a.v_proj(h), a.q_norm.weight, a.k_norm.weight, cos, sin)
        self.k[i].index_copy_(1, pos, k[:, None])
        self.v[i].index_copy_(1, pos, v[:, None])
        # The same attention call as the model's prompt pass (SDPA), so steps round as it does.
        y = F.scaled_dot_product_attention(
            q[None, :, None], self.k[i][None, :, :window], self.v[i][None, :, :window],
            attn_mask=(self.span[:window] <= pos)[None], scale=self.scale, enable_gqa=True,
        ).reshape(1, -1)  # fmt: skip
        return a.o_proj(self.s.gate(y, gate))

    def _moe(self, mlp, h: Tensor) -> Tensor:
        s = self.s
        top, experts = s.route(F.linear(h, mlp.gate.weight))
        down = mlp.experts.routed_down(h, experts[0])
        e = mlp.shared_expert
        shared = e.down_proj(s.swiglu(e.gate_proj(h), e.up_proj(h)))
        return s.combine(down, top[0], shared, mlp.shared_expert_gate(h))


class Decoder:
    def __init__(self, m: Drama3, graphs: bool = True, fused: bool = False):
        self.m, self.graphs, self.fused, self.device = m, graphs, fused, m.device
        dev = self.device
        self.steps = Steps.of(m, fused)
        self.backbone = Backbone(m, self.steps)
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
    def _tower_layer(self, layer, hidden: Tensor, emb: Tensor, pos: Tensor, window: int) -> Tensor:
        """TowerLayer.forward for one position, with the fused steps."""
        s, w = self.steps, layer.w
        mixed = linear(s.tower_in(emb, hidden, w["enorm"], w["hnorm"]), w["eh_proj"])
        cos, sin = layer.cos.index_select(0, pos), layer.sin.index_select(0, pos)
        q, gate, k, v = s.tower_heads(
            linear(s.norm(mixed, w["attention_norm"]), w["wqkv"]), w["q_norm"], w["k_norm"], cos, sin
        )
        layer.k.index_copy_(0, pos, k[None])
        layer.v.index_copy_(0, pos, v[None])
        y = F.scaled_dot_product_attention(
            q[None, :, None], layer.k[:window].transpose(0, 1)[None], layer.v[:window].transpose(0, 1)[None],
            attn_mask=(layer.span[:window] <= pos)[None], enable_gqa=True,
        ).reshape(1, -1)  # fmt: skip
        residual, ff = s.add_norm(linear(s.gate(y, gate), w["wo"]), mixed, w["ffn_norm"])
        ff = linear(s.swiglu(linear(ff, w["w1"]), linear(ff, w["w3"])), w["w2"])
        return s.tower_out(ff, residual, w["final_layernorm"])

    def _tower(self, label: Tensor, window: int) -> Tensor:
        """Drama3.tower for the frame after ``frame`` (at ``pos``): codes [1, 10], 1..9 drawn."""
        m = self.m
        semantic = m.is_semantic(label[:, 0])[:, None]
        frame = self.frame
        shared_in = torch.where(m.is_semantic(frame[:, 0])[:, None], m.code_sum(frame, 2, 10), 0.0)
        h, code0 = self.h, label[:, 1]
        for i, layer in enumerate(m.shared):
            emb = torch.where(semantic, F.embedding(code0.clamp(0, 4095), layer.w["emb"]), 0.0)
            if i == 0:  # the first shared layer also sees the input frame's codes 2..9
                emb = emb + shared_in.to(emb.dtype)
            h = self._tower_layer(layer, h, emb, self.pos, window)
        codes = [code0]
        for layer, head in zip(m.residual, m.heads, strict=True):
            emb = torch.where(semantic, F.embedding(codes[-1].clamp(0, 4095), layer.w["emb"]), 0.0)
            h = self._tower_layer(layer, h, emb, self.pos, window)
            codes.append(pick(linear(h, head)[0].float(), self.temperature, self.top_p, self.top_k))
        return torch.stack(codes, -1)

    def _step(self, window: int) -> None:
        m = self.m
        logits = F.linear(self.h, m.head)[0].float()
        first = m.candidates.index_select(0, pick(logits, self.temperature, self.top_p, self.top_k))[0]
        again = m.candidates.index_select(0, pick(logits, *self.redraw))[0]
        main = torch.where(m.is_semantic(first) & (self.recent == first).any(), again, first)
        self.recent.copy_(torch.cat([self.recent[1:], main.view(1)]))
        code0 = (main - m.semantic_begin).clamp(0, 4095)
        label = torch.cat([main.view(1), code0.view(1), self.no_codes]).view(1, CODES + 1)
        codes = self._tower(label, window)
        frame = torch.cat([main.view(1), codes[0]]).view(1, CODES + 1)
        self.frame.copy_(frame)
        self.pos.add_(1)
        self.h.copy_(self.backbone.step(m.backbone_input(frame, audio_only=True), self.pos, window))

    # ------------------------------------------------------------ graphs
    @property
    def warm(self) -> bool:
        """False until the first frame graph is recorded (it waits for torch.compile)."""
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
