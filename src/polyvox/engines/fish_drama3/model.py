"""Drama 3's model, run from the 4-bit build (convert.py).

- Backbone: Qwen3.5-35B-A3B (40 layers, 3 Gated DeltaNet : 1 gated full attention, 256 routed
  experts of which 8 per token, plus a gated shared expert) through transformers' qwen3_5_moe
  modules (the checkpoint keys are theirs). Its 4-bit attention linears are dequantized to bf16
  once. At audio rows its input is (token embedding + code-0 embedding) / sqrt(2).
- Routed experts (16.6 GB in 4 bits): pinned CPU memory, with a cache of slots per layer on the
  GPU (as many as fit, least recently used out); what the router picks and the cache lacks is
  copied over for the call.
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
from collections import OrderedDict
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


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 30


DEFAULT_SAMPLING = Sampling()  # Fish's defaults


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


class Experts(nn.Module):
    """One layer's routed experts: 4-bit stacks in pinned CPU memory ({gate,up,down}_{q,s},
    [E, ...]) and ``slots`` of them on the GPU."""

    def __init__(self, stacks: dict[str, torch.Tensor], slots: int, device):
        super().__init__()
        self.cpu = stacks
        self.device = device
        self.slots = slots
        self.gpu = {k: torch.empty((slots, *v.shape[1:]), dtype=v.dtype, device=device) for k, v in stacks.items()}
        self.held: OrderedDict[int, int] = OrderedDict()  # expert -> slot, least recently used first
        self.hits = self.misses = 0

    def _take(self, expert: int, keep: set[int]) -> int | None:
        """A slot for ``expert`` (copied in), or None when every slot holds one of ``keep``."""
        if len(self.held) < self.slots:
            slot = len(self.held)
        else:
            victim = next((e for e in self.held if e not in keep), None)
            if victim is None:
                return None
            slot = self.held.pop(victim)
        for k, v in self.cpu.items():
            self.gpu[k][slot].copy_(v[expert], non_blocking=True)
        self.held[expert] = slot
        return slot

    def forward(self, hidden, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden)
        ids = top_k_index.unique().tolist()
        wanted = set(ids)
        cached, streamed = [], []
        for e in ids:
            if e in self.held:
                self.held.move_to_end(e)
                cached.append((e, self.held[e]))
                self.hits += 1
            else:
                self.misses += 1
                slot = self._take(e, wanted) if self.slots else None
                if slot is None:
                    streamed.append(e)
                else:
                    cached.append((e, slot))
        if cached:
            index = torch.tensor([s for _, s in cached], device=self.device)
            self._run({k: v.index_select(0, index) for k, v in self.gpu.items()}, [e for e, _ in cached],
                      hidden, top_k_index, top_k_weights, out)  # fmt: skip
        for i in range(0, len(streamed), STREAM_CHUNK):
            chunk = streamed[i : i + STREAM_CHUNK]
            rows = torch.tensor(chunk)
            parts = {k: v[rows].to(self.device, non_blocking=True) for k, v in self.cpu.items()}
            self._run(parts, chunk, hidden, top_k_index, top_k_weights, out)
        return out

    @staticmethod
    def _run(parts, experts, hidden, top_k_index, top_k_weights, out):
        wg = dequant(parts["gate_q"], parts["gate_s"], hidden.dtype)
        wu = dequant(parts["up_q"], parts["up_s"], hidden.dtype)
        wd = dequant(parts["down_q"], parts["down_s"], hidden.dtype)
        for j, e in enumerate(experts):
            tok, pos = torch.where(top_k_index == e)
            x = hidden[tok]
            y = (F.silu(x @ wg[j]) * (x @ wu[j])) @ wd[j]
            out.index_add_(0, tok, (y * top_k_weights[tok, pos, None]).to(out.dtype))


def gemma_norm(x, w, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)


class TowerLayer(nn.Module):
    """One codebook layer: mixes the code embedding with the previous state, then a gated GQA
    attention (its own cache over the sequence) and a SwiGLU MLP, Gemma-style norms."""

    def __init__(self, w: dict[str, torch.Tensor], heads=16, kv_heads=2, head_dim=256, rotary_dim=64, base=1e7):
        super().__init__()
        self.w, self.h, self.kvh, self.d, self.rd = w, heads, kv_heads, head_dim, rotary_dim
        self.inv = 1.0 / base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        self.k: torch.Tensor | None = None  # [T, kv_heads, head_dim]
        self.v: torch.Tensor | None = None

    def reset(self):
        self.k = self.v = None

    def rotate(self, x, positions):  # interleaved pairs on the first rotary_dim dims; x [T, H, d]
        ang = positions.float()[:, None] * self.inv.to(x.device)
        cos, sin = ang.cos()[:, None], ang.sin()[:, None]
        r = x[..., : self.rd].float()
        even, odd = r[..., 0::2], r[..., 1::2]
        rot = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
        return torch.cat((rot.to(x.dtype), x[..., self.rd :]), -1)

    def forward(self, hidden, emb, positions):
        """hidden [T, dim] (the previous layer's output), emb [T, dim]; appends these T
        positions to the cache and attends causally."""
        w = self.w
        mixed = F.linear(torch.cat((gemma_norm(emb, w["enorm"]), gemma_norm(hidden, w["hnorm"])), -1), w["eh_proj"])
        x = gemma_norm(mixed, w["attention_norm"])
        qw, kw = self.h * self.d, self.kvh * self.d
        q, gate, k, v = F.linear(x, w["wqkv"]).split([qw, qw, kw, kw], -1)
        t = x.shape[0]
        q = self.rotate(gemma_norm(q.view(t, self.h, self.d), w["q_norm"]), positions)
        k = self.rotate(gemma_norm(k.view(t, self.kvh, self.d), w["k_norm"]), positions)
        v = v.view(t, self.kvh, self.d)
        start = 0 if self.k is None else self.k.shape[0]
        self.k = k if self.k is None else torch.cat((self.k, k))
        self.v = v if self.v is None else torch.cat((self.v, v))
        mask = torch.ones(t, self.k.shape[0], dtype=torch.bool, device=x.device).tril(start)
        y = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], self.k.transpose(0, 1)[None], self.v.transpose(0, 1)[None],
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

    # frames [T, 11]: the main token, then codes 0..9
    def is_semantic(self, ids):
        return (ids >= self.semantic_begin) & (ids <= self.semantic_end)

    def code_sum(self, frames, first, end):
        rows = frames[:, 1 + first : 1 + end].clamp(0, 4095) + self.offsets[first:end]
        return F.embedding_bag(rows, self.codebook_embeddings, mode="sum")

    def backbone_input(self, frames):
        text = self.backbone.embed_tokens(frames[:, 0])
        mixed = (text + self.code_sum(frames, 1, 2).to(text.dtype)) / math.sqrt(2.0)
        return torch.where(self.is_semantic(frames[:, 0])[:, None], mixed, text)

    def tower(self, hidden, positions, input_frames, label_frames, sample_fn=None):
        """The 13 tower layers for these positions. ``label_frames`` give the codes each layer
        embeds (teacher forcing); with ``sample_fn`` (one row) codes 1..9 are drawn."""
        semantic = self.is_semantic(label_frames[:, 0])[:, None]
        shared_in = torch.where(self.is_semantic(input_frames[:, 0])[:, None], self.code_sum(input_frames, 2, 10), 0.0)
        h = hidden
        code0 = label_frames[:, 1]
        for i, layer in enumerate(self.shared):
            emb = torch.where(semantic, F.embedding(code0.clamp(0, 4095), layer.w["emb"]), 0.0)
            if i == 0:  # the first shared layer also sees the input frame's codes 2..9
                emb = emb + shared_in.to(emb.dtype)
            h = layer(h, emb, positions)
        codes = [code0]
        for r, (layer, head) in enumerate(zip(self.residual, self.heads, strict=True)):
            emb = torch.where(semantic, F.embedding(codes[-1].clamp(0, 4095), layer.w["emb"]), 0.0)
            h = layer(h, emb, positions)
            codes.append(sample_fn(F.linear(h, head)) if sample_fn is not None else label_frames[:, 2 + r])
        return torch.stack(codes, -1)

    @torch.inference_mode()
    def generate(
        self,
        rows: torch.Tensor,
        sampling: Sampling = DEFAULT_SAMPLING,
        max_frames: int = 1024,
        on_frame: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        from transformers.cache_utils import DynamicCache

        dev = self.device
        rows = rows.to(dev)
        for layer in (*self.shared, *self.residual):
            layer.reset()
        cache = DynamicCache(config=self.text_config)
        n = rows.shape[0]
        hidden = self.backbone(inputs_embeds=self.backbone_input(rows)[None], past_key_values=cache, use_cache=True)
        hidden = hidden.last_hidden_state[0]
        positions = torch.arange(n, device=dev)
        if n > 1:  # the tower over the prompt: row p embeds frame p + 1
            self.tower(hidden[:-1], positions[:-1], rows[:-1], rows[1:])
        t, k, p = sampling.temperature, sampling.top_k, sampling.top_p
        recent = torch.full((RECENT,), -1, device=dev)
        frames, frame_in, pos, h = [], rows[-1:], n - 1, hidden[-1:]
        for _ in range(max_frames):
            logits = F.linear(h, self.head)[0]
            first = self.candidates[sample(logits, t, k, p)]
            again = self.candidates[sample(logits, 1.0, 30, 0.95)]
            semantic = bool(self.is_semantic(first))
            main = again if (semantic and bool((recent == first).any())) else first
            if int(main) == self.eos:
                break
            label = torch.zeros(1, 11, dtype=torch.long, device=dev)
            label[0, 0], label[0, 1] = main, (main - self.semantic_begin).clamp(0, 4095)
            codes = self.tower(
                h, torch.tensor([pos], device=dev), frame_in, label, lambda lg: sample(lg[0], t, k, p)[None]
            )
            frame = torch.cat([main.view(1), codes[0]]).view(1, 11)
            frames.append(frame)
            recent = torch.cat([recent[1:], main.view(1)])
            pos += 1
            frame_in = frame
            out = self.backbone(inputs_embeds=self.backbone_input(frame)[None], past_key_values=cache, use_cache=True)
            h = out.last_hidden_state[0]
            if on_frame:
                on_frame()
        if not frames:
            return torch.zeros(10, 0, dtype=torch.long)
        return torch.cat(frames)[:, 1:].T.cpu()

    def cache_stats(self) -> tuple[int, int]:
        """(hits, misses) of the expert caches so far."""
        layers = [layer.mlp.experts for layer in self.backbone.layers]
        return sum(x.hits for x in layers), sum(x.misses for x in layers)


def load(root: Path, device: str = "cuda", expert_budget: Callable[[int], int] | None = None) -> Drama3:
    """The 4-bit model in ``root``. ``expert_budget(bytes of one expert in every layer)`` -> GPU
    slots per layer for the expert caches, asked once everything else is on the GPU (None: no cache)."""
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

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
    state = {}
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
            w = dequant(get(key).to(device), get(base + ".scales").to(device))  # [K, N]
            state[name.removesuffix(".qweight") + ".weight"] = w.T.contiguous()
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
    m.offsets = torch.arange(10, device=device) * 4096

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
    slots = expert_budget(expert_bytes(stacks[0]) * len(stacks)) if expert_budget else 0
    slots = max(0, min(slots, m.text_config.num_experts))
    for layer, parts in zip(m.backbone.layers, stacks, strict=True):
        layer.mlp.experts = Experts(parts, slots, device)
    logger.info("Drama 3 loaded in %.0f s; %d experts per layer cached on the GPU", time.time() - t0, slots)
    return m
