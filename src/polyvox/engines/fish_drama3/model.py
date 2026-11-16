"""Drama 3's model, run from the 4-bit build (convert.py).

- Backbone: Qwen3.5-35B-A3B (40 layers, 3 Gated DeltaNet : 1 gated full attention, 256 routed
  experts of which 8 per token, plus a gated shared expert). The prompt goes through
  transformers' qwen3_5_moe modules (the checkpoint keys are theirs); decoding is decode.py's
  own step on the same weights. The 4-bit attention linears are dequantized to bf16 once. At
  audio rows its input is (token embedding + code-0 embedding) / sqrt(2).
- Routed experts (16.6 GB in 4 bits): pinned CPU memory, the most used of them also in GPU slots
  (as many as fit). A decoding step reads each picked expert from its slot, or straight from
  CPU memory over PCIe (kernels.py), so nothing waits on the CPU; ``plan`` refreshes which
  experts are held from their use (the prompt's, then the frames').
- Tower: 4 shared + 9 residual layers predicting codes 1..9 of each frame from the backbone's
  state (layer r embeds code r; its head gives code r + 1).
- Sampling: temperature / top-p / top-k; a semantic token seen in the last 16 frames is drawn
  again at 1.0 / 0.95 / 30 (repetition-aware sampling).

Ported from sglang-omni (Apache-2.0, sglang_omni/models/fishaudio_drama3).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .convert import dequant

logger = logging.getLogger(__name__)

# The config keys transformers' Qwen3_5MoeTextConfig doesn't take.
FISH_ONLY = ("model_type", "mlp_layer_types", "rotary_interleaved", "use_qk_norm",
             "layernorm_zero_centered_gamma", "shared_expert_gate", "attn_output_gate")  # fmt: skip
DENSE = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj")
PARTS = (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down"))
STREAM_CHUNK = 48  # experts copied at a time when the cache can't hold a call's
RECENT = 16  # frames a repeated semantic token is looked for in
CAPACITY = 4096  # sequence positions: references, text and 1024 frames
CODES, CODEBOOK = 10, 4096
PREFIXES = 2  # voices whose prompt states are kept
ROUTED_TOKENS = 48  # up to this many tokens, experts are read per token (no layer-wide copies)


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 30


DEFAULT_SAMPLING = Sampling()  # Fish's defaults
REDRAW = Sampling(1.0, 0.95, 30)  # a repeated semantic token is drawn again with these


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    x = logits.float() / max(temperature, 1e-5)
    srt, order = x.sort(-1, descending=True)
    cum = srt.softmax(-1).cumsum(-1)
    rank = torch.arange(srt.shape[-1], device=x.device)
    drop = (cum > top_p) | (rank >= top_k)
    drop[..., 0] = False
    p = srt.masked_fill(drop, float("-inf")).softmax(-1)
    choice = (p / torch.empty_like(p).exponential_(1)).argmax(-1, keepdim=True)
    return order.gather(-1, choice).squeeze(-1)


_STAGING: dict[str, torch.Tensor] = {}
_WHOLE: dict[str, torch.Tensor] = {}


def _staging(name: str, like: torch.Tensor) -> torch.Tensor:
    """A pinned buffer of STREAM_CHUNK experts shaped like ``like`` (shared by all layers)."""
    buf = _STAGING.get(name)
    shape = (STREAM_CHUNK, *like.shape[1:])
    if buf is None or buf.shape != shape or buf.dtype != like.dtype:
        buf = torch.empty(shape, dtype=like.dtype, pin_memory=torch.cuda.is_available())
        _STAGING[name] = buf
    return buf


def _on_gpu(stacks: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    """A layer's whole expert stacks copied into GPU buffers shared by all layers (stream-ordered:
    the next layer's copy waits for this one's work)."""
    out = {}
    for k, v in stacks.items():
        buf = _WHOLE.get(k)
        if buf is None or buf.shape != v.shape or buf.dtype != v.dtype or buf.device != torch.device(device):
            buf = _WHOLE[k] = torch.empty_like(v, device=device)
        out[k] = buf.copy_(v, non_blocking=True)
    return out


def release_buffers() -> None:
    """Free the shared staging buffers (unloading the model)."""
    _WHOLE.clear()
    _STAGING.clear()


class Experts(nn.Module):
    """One layer's routed experts: 4-bit stacks in pinned CPU memory ({gate,up,down}_{q,s},
    [E, ...]), ``slots`` of them also on the GPU (``table``: an expert's slot, or -1)."""

    def __init__(self, stacks: dict[str, torch.Tensor], slots: int, device):
        super().__init__()
        self.cpu = stacks
        self.device = device
        self.slots = slots
        self.experts = next(iter(stacks.values())).shape[0]
        self.gpu = {k: torch.empty((slots, *v.shape[1:]), dtype=v.dtype, device=device) for k, v in stacks.items()}
        self.held: dict[int, int] = {}  # expert -> slot
        self.table = torch.full((self.experts,), -1, dtype=torch.int32, device=device)
        self.counts = torch.zeros(self.experts, dtype=torch.float32, device=device)  # use since the last plan
        self.fast = str(device).startswith("cuda")
        if self.fast:
            from .kernels import HostPointer

            self.host = {k: HostPointer(v) for k, v in stacks.items()}
        self.rows = torch.arange(64, dtype=torch.int32, device=device)
        self.zeros = torch.zeros(64, dtype=torch.int32, device=device)

    def plan(self, decay: float = 0.5) -> float:
        """Hold the most used experts in the GPU slots (outside a graph: it reads the counts).
        Returns the share of the use the held experts cover."""
        if not self.slots:
            return 0.0
        counts = self.counts.cpu()
        total = float(counts.sum())
        want = counts.topk(min(self.slots, self.experts)).indices.tolist()
        keep = {e: s for e, s in self.held.items() if e in set(want)}
        free = [s for s in range(self.slots) if s not in set(keep.values())]
        for e in want:
            if e not in keep:
                slot = free.pop()
                for k, v in self.cpu.items():
                    self.gpu[k][slot].copy_(v[e], non_blocking=True)
                keep[e] = slot
        self.held = keep
        table = torch.full((self.experts,), -1, dtype=torch.int32)
        table[list(keep)] = torch.tensor(list(keep.values()), dtype=torch.int32)
        self.table.copy_(table)
        covered = float(counts[list(keep)].sum()) / total if total else 0.0
        self.counts.mul_(decay)
        return covered

    def forward(self, hidden, top_k_index, top_k_weights):
        """Any number of tokens (the prompt): rows grouped by expert, the held experts from their
        slots and the rest streamed from CPU memory."""
        if hidden.shape[0] == 1:
            return self.routed(hidden, top_k_index[0], top_k_weights[0])
        if self.fast and hidden.shape[0] <= ROUTED_TOKENS:  # a text after kept references: per token
            return self.routed_tokens(hidden, top_k_index, top_k_weights)
        out = torch.zeros_like(hidden)
        flat = top_k_index.flatten()
        self.counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        order = flat.argsort(stable=True)
        counts = torch.bincount(flat, minlength=self.experts).tolist()  # one sync for the layer
        tokens, gates = order // top_k_index.shape[1], top_k_weights.flatten()[order]
        groups, start = {}, 0
        for e, c in enumerate(counts):
            if c:
                groups[e] = (tokens[start : start + c], gates[start : start + c])
                start += c
        cached = [(e, self.held[e]) for e in groups if e in self.held]
        streamed = [e for e in groups if e not in self.held]
        if cached:
            index = torch.tensor([s for _, s in cached], device=self.device)
            self._run({k: v.index_select(0, index) for k, v in self.gpu.items()}, [e for e, _ in cached],
                      hidden, groups, out)  # fmt: skip
        # Many missing (a prompt needs nearly every expert): the layer's whole stack to the GPU at once.
        whole = _on_gpu(self.cpu, self.device) if len(streamed) > STREAM_CHUNK else None
        for i in range(0, len(streamed), STREAM_CHUNK):
            chunk = streamed[i : i + STREAM_CHUNK]
            rows = torch.tensor(chunk)
            parts = {}
            for k, v in self.cpu.items():
                if whole is not None:
                    parts[k] = whole[k].index_select(0, rows.to(self.device))
                else:  # gathered into pinned memory: a fast copy (indexing makes pageable)
                    stage = _staging(k, v)[: len(chunk)]
                    torch.index_select(v, 0, rows, out=stage)
                    parts[k] = stage.to(self.device)  # waits: the staging buffer is reused
            self._run(parts, chunk, hidden, groups, out)
        return out

    def routed(self, hidden, experts, weights):
        """One token [1, dim] through its experts [k] with their weights [k]. On CUDA nothing is
        decided on the CPU (graph-safe): each expert is read from its slot or from CPU memory."""
        down = self.routed_down(hidden, experts)
        return (down * weights.float()[:, None]).sum(0, keepdim=True).to(hidden.dtype)

    def routed_down(self, hidden, experts):
        """routed's expert outputs before weighting: [k, dim] float32 (and their use counted)."""
        k = experts.shape[0]
        self.counts.index_add_(0, experts, torch.ones(k, dtype=torch.float32, device=experts.device))
        if self.fast:
            from .kernels import int4_gemv_routed

            ids = experts.to(torch.int32)
            g, h = self.gpu, self.host
            gate = int4_gemv_routed(
                hidden, self.zeros[:k], ids, self.table, g["gate_q"], g["gate_s"], h["gate_q"], h["gate_s"]
            )
            up = int4_gemv_routed(hidden, self.zeros[:k], ids, self.table, g["up_q"], g["up_s"], h["up_q"], h["up_s"])
            act = (F.silu(gate) * up).to(hidden.dtype)
            down = int4_gemv_routed(
                act, self.rows[:k], ids, self.table, g["down_q"], g["down_s"], h["down_q"], h["down_s"]
            )
        else:  # reference path (CPU): the experts' weights unpacked
            picked = experts.cpu()
            w = {n: dequant(self.cpu[n + "_q"][picked], self.cpu[n + "_s"][picked], torch.float32).to(hidden.device)
                 for n in ("gate", "up", "down")}  # fmt: skip
            x = hidden[0].float()
            gate, up = torch.einsum("d,edi->ei", x, w["gate"]), torch.einsum("d,edi->ei", x, w["up"])
            act = (F.silu(gate) * up).to(hidden.dtype).float()
            down = torch.einsum("ei,eid->ed", act, w["down"])
        return down

    def routed_tokens(self, hidden, top_k_index, top_k_weights):
        """A few tokens [T, dim]: each (token, expert) pair read as in ``routed`` (CUDA)."""
        from .kernels import int4_gemv_routed

        t, k = top_k_index.shape
        ids = top_k_index.flatten()
        self.counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
        ids = ids.to(torch.int32)
        tokens = torch.arange(t, dtype=torch.int32, device=hidden.device).repeat_interleave(k)
        rows = torch.arange(t * k, dtype=torch.int32, device=hidden.device)
        g, h = self.gpu, self.host
        gate = int4_gemv_routed(hidden, tokens, ids, self.table, g["gate_q"], g["gate_s"], h["gate_q"], h["gate_s"])
        up = int4_gemv_routed(hidden, tokens, ids, self.table, g["up_q"], g["up_s"], h["up_q"], h["up_s"])
        act = (F.silu(gate) * up).to(hidden.dtype)
        down = int4_gemv_routed(act, rows, ids, self.table, g["down_q"], g["down_s"], h["down_q"], h["down_s"])
        return (down.view(t, k, -1) * top_k_weights.float()[:, :, None]).sum(1).to(hidden.dtype)

    @staticmethod
    def _run(parts, experts, hidden, groups, out):
        """``groups``: expert -> (its rows of ``hidden``, their router weights)."""
        if hidden.is_cuda and hidden.dtype == torch.bfloat16:
            from .kernels import int4_dequant as unpack
        else:

            def unpack(q, s):
                return dequant(q, s, hidden.dtype)

        wg = unpack(parts["gate_q"], parts["gate_s"])
        wu = unpack(parts["up_q"], parts["up_s"])
        wd = unpack(parts["down_q"], parts["down_s"])
        for j, e in enumerate(experts):
            tok, gate = groups[e]
            x = hidden[tok]
            y = (F.silu(x @ wg[j]) * (x @ wu[j])) @ wd[j]
            out.index_add_(0, tok, (y * gate[:, None]).to(out.dtype))


def gemma_norm(x, w, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)


@dataclass(frozen=True)
class TowerShape:
    heads: int = 16
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    base: float = 1e7


DRAMA3_TOWER = TowerShape()


class TowerLayer(nn.Module):
    """One codebook layer: mixes the code embedding with the previous state, then a gated GQA
    attention (its own cache over the sequence) and a SwiGLU MLP, Gemma-style norms."""

    def __init__(self, w: dict[str, torch.Tensor], shape: TowerShape = DRAMA3_TOWER, capacity: int = CAPACITY):
        super().__init__()
        s = self.shape = shape
        self.w = w
        dev, dtype = w["wqkv"].device, w["wqkv"].dtype
        inv = 1.0 / s.base ** (torch.arange(0, s.rotary_dim, 2, dtype=torch.float32, device=dev) / s.rotary_dim)
        ang = torch.arange(capacity, dtype=torch.float32, device=dev)[:, None] * inv
        self.cos, self.sin = ang.cos(), ang.sin()  # [capacity, rotary_dim / 2]
        self.k = torch.zeros(capacity, s.kv_heads, s.head_dim, dtype=dtype, device=dev)
        self.v = torch.zeros_like(self.k)
        self.span = torch.arange(capacity, device=dev)

    def rotate(self, x, positions):  # interleaved pairs on the first rotary_dim dims; x [T, H, d]
        rd = self.shape.rotary_dim
        cos, sin = self.cos.index_select(0, positions)[:, None], self.sin.index_select(0, positions)[:, None]
        r = x[..., :rd].float()
        even, odd = r[..., 0::2], r[..., 1::2]
        rot = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
        return torch.cat((rot.to(x.dtype), x[..., rd:]), -1)

    def forward(self, hidden, emb, positions, window: int):
        """hidden [T, dim] (the previous layer's output), emb [T, dim], at ``positions`` [T]
        (written to the cache); attends over the first ``window`` positions, causally."""
        w, s = self.w, self.shape
        mixed = F.linear(torch.cat((gemma_norm(emb, w["enorm"]), gemma_norm(hidden, w["hnorm"])), -1), w["eh_proj"])
        x = gemma_norm(mixed, w["attention_norm"])
        qw, kw = s.heads * s.head_dim, s.kv_heads * s.head_dim
        q, gate, k, v = F.linear(x, w["wqkv"]).split([qw, qw, kw, kw], -1)
        t = x.shape[0]
        q = self.rotate(gemma_norm(q.view(t, s.heads, s.head_dim), w["q_norm"]), positions)
        k = self.rotate(gemma_norm(k.view(t, s.kv_heads, s.head_dim), w["k_norm"]), positions)
        self.k.index_copy_(0, positions, k)
        self.v.index_copy_(0, positions, v.view(t, s.kv_heads, s.head_dim))
        mask = self.span[:window][None] <= positions[:, None]
        y = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], self.k[:window].transpose(0, 1)[None], self.v[:window].transpose(0, 1)[None],
            attn_mask=mask, enable_gqa=True,
        )[0].transpose(0, 1).reshape(t, qw)  # fmt: skip
        residual = F.linear(y * torch.sigmoid(gate), w["wo"]) + mixed
        ff = gemma_norm(residual, w["ffn_norm"])
        ff = F.linear(F.silu(F.linear(ff, w["w1"])) * F.linear(ff, w["w3"]), w["w2"])
        return gemma_norm(ff + residual, w["final_layernorm"])


def expert_bytes(stacks: dict[str, torch.Tensor]) -> int:
    """GPU bytes of one expert of a layer."""
    return sum(v[0].numel() * v.element_size() for v in stacks.values())


class Drama3(nn.Module):
    """The model (``load``); ``generate`` speaks a prompt (prompt.build) as codes [10, T]."""

    def __init__(self):
        super().__init__()
        self.decoder = None
        self.prefixes: dict[str, tuple] = {}  # a voice's prompt part -> the states after it (prefill)

    # frames [T, 11]: the main token, then codes 0..9
    def is_semantic(self, ids):
        return (ids >= self.semantic_begin) & (ids <= self.semantic_end)

    def code_sum(self, frames, first, end):
        rows = frames[:, 1 + first : 1 + end].clamp(0, CODEBOOK - 1) + self.offsets[first:end]
        return F.embedding_bag(rows, self.codebook_embeddings, mode="sum")

    def backbone_input(self, frames):
        text = self.backbone.embed_tokens(frames[:, 0])
        mixed = (text + self.code_sum(frames, 1, 2).to(text.dtype)) / math.sqrt(2.0)
        return torch.where(self.is_semantic(frames[:, 0])[:, None], mixed, text)

    def tower(self, hidden, positions, input_frames, label_frames, window: int, sample_fn=None):
        """The 13 tower layers at ``positions``. ``label_frames`` give the codes each layer embeds
        (teacher forcing); with ``sample_fn`` (one row) codes 1..9 are drawn."""
        semantic = self.is_semantic(label_frames[:, 0])[:, None]
        shared_in = torch.where(self.is_semantic(input_frames[:, 0])[:, None], self.code_sum(input_frames, 2, 10), 0.0)
        h = hidden
        code0 = label_frames[:, 1]
        for i, layer in enumerate(self.shared):
            emb = torch.where(semantic, F.embedding(code0.clamp(0, CODEBOOK - 1), layer.w["emb"]), 0.0)
            if i == 0:  # the first shared layer also sees the input frame's codes 2..9
                emb = emb + shared_in.to(emb.dtype)
            h = layer(h, emb, positions, window)
        codes = [code0]
        for r, (layer, head) in enumerate(zip(self.residual, self.heads, strict=True)):
            emb = torch.where(semantic, F.embedding(codes[-1].clamp(0, CODEBOOK - 1), layer.w["emb"]), 0.0)
            h = layer(h, emb, positions, window)
            codes.append(sample_fn(F.linear(h, head)) if sample_fn is not None else label_frames[:, 2 + r])
        return torch.stack(codes, -1)

    def _pass(self, rows, start: int, cache):
        """Rows at positions start.. through the backbone (``cache`` holds the ones before):
        their hidden states [T, dim]."""
        positions = torch.arange(start, start + rows.shape[0], device=self.device)[None]
        out = self.backbone(
            inputs_embeds=self.backbone_input(rows)[None], position_ids=positions, past_key_values=cache, use_cache=True
        )
        return out.last_hidden_state[0]

    def prefill(self, rows: torch.Tensor, head: int = 0):
        """The prompt through the backbone (transformers' modules) and the tower; returns
        (the last row's hidden state [1, dim], the backbone's cache). The state after the first
        ``head`` rows (a voice's references: the same for each text it speaks) is kept, so the
        next prompt with the same ones starts after them."""
        import copy
        import hashlib

        from transformers.cache_utils import DynamicCache

        n = rows.shape[0]
        if n + 2 > CAPACITY:
            raise ValueError(f"The prompt is too long for Drama 3 ({n} positions).")
        towers = (*self.shared, *self.residual)
        if not 1 < head < n:
            cache = DynamicCache(config=self.text_config)
            hidden = self._pass(rows, 0, cache)
            if n > 1:  # the tower over the prompt: row p embeds frame p + 1
                self.tower(hidden[:-1], torch.arange(n - 1, device=self.device), rows[:-1], rows[1:], n - 1)
            return hidden[-1:], cache
        key = hashlib.sha1(rows[:head].cpu().numpy().tobytes()).hexdigest()
        prefix = self.prefixes.pop(key, None)
        if prefix is None:
            cache = DynamicCache(config=self.text_config)
            hidden = self._pass(rows[:head], 0, cache)
            self.tower(
                hidden[:-1], torch.arange(head - 1, device=self.device), rows[: head - 1], rows[1:head], head - 1
            )
            kv = [(t.k[: head - 1].clone(), t.v[: head - 1].clone()) for t in towers]
            prefix = (cache, hidden[-1:].clone(), kv)
        else:
            for t, (k, v) in zip(towers, prefix[2], strict=True):
                t.k[: head - 1].copy_(k)
                t.v[: head - 1].copy_(v)
        self.prefixes[key] = prefix  # most recent last
        while len(self.prefixes) > PREFIXES:
            self.prefixes.pop(next(iter(self.prefixes)))
        cache = copy.deepcopy(prefix[0])
        hidden = self._pass(rows[head:], head, cache)
        before = torch.cat((prefix[1], hidden[:-1]))  # the hidden states at positions head - 1 .. n - 2
        self.tower(before, torch.arange(head - 1, n - 1, device=self.device), rows[head - 1 : -1], rows[head:], n - 1)
        return hidden[-1:], cache

    def plan_experts(self) -> float:
        """Refresh the experts held on the GPU from their use; the mean share of use they cover."""
        shares = [layer.mlp.experts.plan() for layer in self.backbone.layers]
        return sum(shares) / len(shares)

    @torch.inference_mode()
    def generate(
        self,
        rows: torch.Tensor,
        sampling: Sampling = DEFAULT_SAMPLING,
        max_frames: int = 1024,
        on_frame: Callable[[], None] | None = None,
        head: int = 0,
    ) -> torch.Tensor:
        """``head``: the rows the references make up (prompt.parts): their states are kept."""
        from .decode import Decoder

        try:
            return self.decoder.generate(rows, sampling, max_frames, on_frame, head)
        except Exception as e:  # fusing failed (torch.compile: no working Triton, say): decode unfused
            if not self.decoder.fused or isinstance(e, torch.OutOfMemoryError):
                raise
            logger.warning("Drama 3: fusing the decoding steps failed (%s); decoding unfused", e)
            graphs = self.decoder.graphs
            self.decoder = None
            self.decoder = Decoder(self, graphs=graphs)
            return self.decoder.generate(rows, sampling, max_frames, on_frame, head)

    @torch.inference_mode()
    def generate_reference(
        self, rows, sampling: Sampling = DEFAULT_SAMPLING, max_frames: int = 1024, head: int = 0
    ) -> torch.Tensor:
        """Plain decoding through transformers' modules (tests compare the decoder with it)."""
        dev = self.device
        rows = rows.to(dev)
        n = rows.shape[0]
        h, cache = self.prefill(rows, head)
        t, k, p = sampling.temperature, sampling.top_k, sampling.top_p
        recent = torch.full((RECENT,), -1, device=dev)
        frames, frame_in, pos = [], rows[-1:], n - 1
        for _ in range(max_frames):
            logits = F.linear(h, self.head)[0]
            first = self.candidates[sample(logits, t, k, p)]
            again = self.candidates[sample(logits, REDRAW.temperature, REDRAW.top_k, REDRAW.top_p)]
            main = again if (bool(self.is_semantic(first)) and bool((recent == first).any())) else first
            if int(main) == self.eos:
                break
            label = torch.zeros(1, 11, dtype=torch.long, device=dev)
            label[0, 0], label[0, 1] = main, (main - self.semantic_begin).clamp(0, CODEBOOK - 1)
            positions = torch.tensor([pos], device=dev)
            codes = self.tower(h, positions, frame_in, label, pos + 1, lambda lg: sample(lg[0], t, k, p)[None])
            frame = torch.cat([main.view(1), codes[0]]).view(1, 11)
            frames.append(frame)
            recent = torch.cat([recent[1:], main.view(1)])
            pos += 1
            frame_in = frame
            out = self.backbone(inputs_embeds=self.backbone_input(frame)[None], past_key_values=cache, use_cache=True)
            h = out.last_hidden_state[0]
        if not frames:
            return torch.zeros(CODES, 0, dtype=torch.long)
        return torch.cat(frames)[:, 1:].T.cpu()


def load(
    root: Path, device: str = "cuda", expert_budget: Callable[[int], int] | None = None, graphs: bool = True
) -> Drama3:
    """The 4-bit model in ``root``. ``expert_budget(bytes of one expert in every layer)`` -> GPU
    slots per layer for the experts, asked once everything else is on the GPU (None: no slots)."""
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    from .decode import Decoder

    t0 = time.time()
    m = Drama3()
    m.device = device
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if (cfg.get("quantization_config") or {}).get("quant_method") != "drama3-int4":
        raise ValueError(f"{root} is not Drama 3's 4-bit build.")
    m.text_config = Qwen3_5MoeTextConfig(**{k: v for k, v in cfg["text_config"].items() if k not in FISH_ONLY})
    m.text_config._attn_implementation = "sdpa"
    m.semantic_begin, m.semantic_end = cfg["semantic_start_token_id"], cfg["semantic_end_token_id"]
    m.eos = cfg["eos_token_id"]
    m.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    weight_map = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    handles: dict[str, object] = {}

    def get(key):
        shard = weight_map[key]
        if shard not in handles:
            handles[shard] = safe_open(str(root / shard), "pt")
        return handles[shard].get_tensor(key)

    with torch.device("meta"):
        m.backbone = Qwen3_5MoeTextModel(m.text_config)
    prefix = "model.language_model."
    state, packed = {}, {}
    on_cuda = str(device).startswith("cuda")
    for key in weight_map:
        if not key.startswith(prefix) or ".experts." in key:
            continue
        name = key.removeprefix(prefix)
        if name.endswith((".g_idx", ".qzeros", ".scales")):
            continue
        if name.endswith(".qweight"):
            base = key.removesuffix(".qweight")
            if not any(base.endswith(d) for d in DENSE):
                raise ValueError(f"unexpected 4-bit tensor {key}")
            qweight, scales = get(key).to(device), get(base + ".scales").to(device)
            if on_cuda:  # kept in 4 bits (kernels.Int4Linear)
                packed[name.removesuffix(".qweight")] = (qweight, scales)
            else:
                state[name.removesuffix(".qweight") + ".weight"] = dequant(qweight, scales).T.contiguous()
        else:
            state[name] = get(key).to(device)
    stacks = []
    for i in range(len(m.backbone.layers)):
        parts = {}
        for proj, short in PARTS:
            for kind, suffix in (("q", "qweight"), ("s", "scales")):
                keys = [f"{prefix}layers.{i}.mlp.experts.{e}.{proj}.{suffix}" for e in range(m.text_config.num_experts)]
                parts[f"{short}_{kind}"] = torch.stack([get(k) for k in keys]).pin_memory()
        stacks.append(parts)
    if packed:
        from .kernels import Int4Linear

        for name, (qweight, scales) in packed.items():
            parent, attr = name.rsplit(".", 1)
            setattr(m.backbone.get_submodule(parent), attr, Int4Linear(qweight, scales))
    missing, unexpected = m.backbone.load_state_dict(state, strict=False, assign=True)
    missing = [k for k in missing if ".experts." not in k]
    if missing or unexpected:
        raise ValueError(f"Drama 3 weights don't fit the backbone: {missing[:3]} {unexpected[:3]}")
    m.backbone.rotary_emb = type(m.backbone.rotary_emb)(m.text_config, device=device)
    m.backbone.eval().requires_grad_(False)
    # Only the frame's possible main tokens of the vocabulary: the semantic codes and the end.
    m.candidates = torch.tensor([*range(m.semantic_begin, m.semantic_end + 1), m.eos], device=device)
    m.head = get("lm_head.weight").to(device)[m.candidates].contiguous()
    m.codebook_embeddings = get("audio_decoder.codebook_embeddings.weight").to(device)
    m.offsets = torch.arange(CODES, device=device) * CODEBOOK

    def tower_weights(p, emb_name):
        tl = p + "transformer_layer."
        keys = {
            "emb": p + emb_name, "enorm": p + "enorm.weight", "hnorm": p + "hnorm.weight",
            "eh_proj": p + "eh_proj.weight", "final_layernorm": p + "final_layernorm.weight",
            "attention_norm": tl + "attention_norm.weight", "ffn_norm": tl + "ffn_norm.weight",
            "wqkv": tl + "attention.wqkv.weight", "wo": tl + "attention.wo.weight",
            "q_norm": tl + "attention.q_norm.weight", "k_norm": tl + "attention.k_norm.weight",
            "w1": tl + "feed_forward.w1.weight", "w2": tl + "feed_forward.w2.weight", "w3": tl + "feed_forward.w3.weight",
        }  # fmt: skip
        return {name: get(key).to(device) for name, key in keys.items()}

    m.shared = [
        TowerLayer(tower_weights(f"audio_decoder_mtp_shared.layers.{i}.", "semantic_embedding.weight"))
        for i in range(4)
    ]
    m.residual = [
        TowerLayer(tower_weights(f"audio_decoder_mtp.layers.{i}.", "codebook_embedding.weight")) for i in range(9)
    ]
    m.heads = [get(f"audio_decoder_mtp.heads.{i}.weight").to(device) for i in range(9)]
    handles.clear()
    graphs = graphs and on_cuda
    # The small steps between the linear layers fused by torch.compile (with the graphs).
    m.decoder = Decoder(m, graphs=graphs, fused=graphs)
    slots = expert_budget(expert_bytes(stacks[0]) * len(stacks)) if expert_budget else 0
    slots = max(0, min(slots, m.text_config.num_experts))
    for layer, parts in zip(m.backbone.layers, stacks, strict=True):
        layer.mlp.experts = Experts(parts, slots, device)
    logger.info("Drama 3 loaded in %.0f s; %d experts per layer held on the GPU", time.time() - t0, slots)
    return m
