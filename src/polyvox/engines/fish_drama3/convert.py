"""The official Drama 3 checkpoint (block-FP8 routed experts, bf16 elsewhere) -> 4-bit weights.

Quantized to INT4 (symmetric, one scale per 128 weights along the input dimension, round to
nearest after a search over 20 clip ratios per group; no calibration data):
- the routed experts (FP8 with 128 x 128 block scales, dequantized first);
- the full-attention q / k / v / o and linear-attention in_proj_qkv / in_proj_z / out_proj.
The shared experts stay bf16 (4 bits there breaks the model: its gate and up errors multiply).
Everything else is copied unchanged.

Layout as the community drama-3-int4 builds (sglang-omni's GPTQ readers load it): experts as
``qweight`` int32 [K/8, N] (value k at bits 4 * (k % 8) of word k // 8, stored + 8) and
``scales`` fp16 [K/128, N]; dense linears also get ``qzeros`` (all 8) and a sorted ``g_idx``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

GROUP = 128
CLIPS = (1.0, 0.81, 20)  # torch.linspace: amax / 7 down to 0.81 x that
EXPERT = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")
DENSE = re.compile(r"\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj))\.weight$")
DENSE_TARGETS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
]  # fmt: skip
ZERO_POINTS = -2004318072  # 0x88888888: eight 4-bit zero points of 8
SHARD_BYTES = 2_000_000_000
STATS = "quant_stats.json"  # written last: a folder with it is complete


def dequant_fp8(w, s):
    """FP8 [N, K] with block scales [ceil(N/128), ceil(K/128)] -> float32 [N, K]."""
    n, k = w.shape
    s = s.float().repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    return w.float() * s


def quantize(w, group: int = GROUP):
    """float32 [N, K] -> (q uint8 [N, K] in 0..15, scales float32 [N, K / group])."""
    import torch

    n, k = w.shape
    wg = w.view(n, k // group, group)
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_err = torch.full_like(amax, float("inf"))
    best_s = amax / 7.0
    for c in torch.linspace(*CLIPS).tolist():
        s = amax * c / 7.0
        err = (((wg / s).round().clamp(-8, 7) * s - wg) ** 2).sum(-1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
    q = (wg / best_s).round().clamp(-8, 7) + 8
    return q.view(n, k).to(torch.uint8), best_s.squeeze(-1)


def restore(q, s):
    """quantize's output -> float32 [N, K]."""
    group = q.shape[1] // s.shape[1]
    return (q.float() - 8) * s.float().repeat_interleave(group, 1)


def pack(q):
    """q uint8 [N, K] (0..15) -> int32 [K/8, N]."""
    import torch

    n, k = q.shape
    qt = q.t().contiguous().view(k // 8, 8, n).to(torch.int64)
    shifts = (torch.arange(8, device=q.device, dtype=torch.int64) * 4).view(1, 8, 1)
    word = (qt << shifts).sum(1)
    return (word - (word >= 2**31).to(torch.int64) * 2**32).to(torch.int32)


def unpack(qweight):
    """int32 [..., K/8, N] -> [..., K, N] in 0..15 (int32)."""
    import torch

    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    return ((qweight.unsqueeze(-2) >> shifts.view(8, 1)) & 0xF).flatten(-3, -2)


def dequant(qweight, scales, dtype=None):
    """int32 [..., K/8, N] + fp16 [..., K/G, N] -> W [..., K, N] = (q - 8) x scale."""
    import torch

    group = qweight.shape[-2] * 8 // scales.shape[-2]
    w = (unpack(qweight) - 8).to(torch.float32) * scales.to(torch.float32).repeat_interleave(group, dim=-2)
    return w.to(dtype or torch.bfloat16)


def convert(src: Path, dst: Path, progress: Callable[[float], None] | None = None, device: str | None = None) -> dict:
    """Write the 4-bit model of ``src`` into ``dst`` (built in ``dst.build`` and renamed when
    complete). ``progress(fraction)`` may raise to cancel. Returns the quantization error stats."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    progress = progress or (lambda fraction: None)
    weight_map = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    handles: dict[str, object] = {}

    def get(key):
        shard = weight_map[key]
        if shard not in handles:
            handles[shard] = safe_open(str(src / shard), "pt")
        return handles[shard].get_tensor(key)

    work = dst.with_name(dst.name + ".build")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    stats: dict[str, dict] = {}
    out: dict = {}
    out_bytes = 0
    names: list[str] = []
    placed: dict[str, str] = {}

    def flush():
        nonlocal out, out_bytes
        if out:
            name = f"model-{len(names) + 1:05d}.safetensors"
            save_file(out, str(work / name), metadata={"format": "pt"})
            names.append(name)
            placed.update(dict.fromkeys(out, name))
            out, out_bytes = {}, 0

    def put(key, t):
        nonlocal out_bytes
        out[key] = t.contiguous().cpu()
        out_bytes += t.numel() * t.element_size()
        if out_bytes > SHARD_BYTES:
            flush()

    def record(kind, w, q, s, src_bytes, int4_bytes):
        err = ((restore(q, s) - w).norm() / w.norm().clamp_min(1e-12)).item()
        st = stats.setdefault(
            kind, {"tensors": 0, "rel_err_sum": 0.0, "rel_err_max": 0.0, "src_bytes": 0, "int4_bytes": 0}
        )
        st["tensors"] += 1
        st["rel_err_sum"] += err
        st["rel_err_max"] = max(st["rel_err_max"], err)
        st["src_bytes"] += src_bytes
        st["int4_bytes"] += int4_bytes

    keys = list(weight_map)
    with torch.inference_mode():
        for i, key in enumerate(keys):
            progress(i / len(keys))
            if key.endswith(".weight_scale_inv") and EXPERT.search(key.removesuffix("_scale_inv")):
                continue  # used with its weight
            base = key.removesuffix(".weight")
            if EXPERT.search(key):
                raw, block = get(key), get(key + "_scale_inv")
                if raw.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"{key}: expected FP8 experts, found {raw.dtype}")
                w = dequant_fp8(raw.to(device), block.to(device))
                q, s = quantize(w)
                qw, sc = pack(q), s.t().contiguous().half()
                record("experts", w, q, s, raw.numel() + block.numel() * 4, qw.numel() * 4 + sc.numel() * 2)
                put(base + ".qweight", qw)
                put(base + ".scales", sc)
            elif DENSE.search(key):
                raw = get(key)
                if raw.dtype != torch.bfloat16 or raw.shape[1] % GROUP:
                    raise ValueError(f"{key}: unexpected {raw.dtype} {tuple(raw.shape)}")
                w = raw.to(device).float()
                q, s = quantize(w)
                n, k = w.shape
                qw, sc = pack(q), s.t().contiguous().half()
                qz = torch.full((k // GROUP, math.ceil(n / 8)), ZERO_POINTS, dtype=torch.int32)
                gi = torch.arange(k, dtype=torch.int32) // GROUP
                record(
                    "dense", w, q, s, raw.numel() * 2, qw.numel() * 4 + qz.numel() * 4 + sc.numel() * 2 + gi.numel() * 4
                )
                put(base + ".qweight", qw)
                put(base + ".qzeros", qz)
                put(base + ".scales", sc)
                put(base + ".g_idx", gi)
            else:
                t = get(key)
                if t.dtype == torch.float8_e4m3fn:
                    raise ValueError(f"{key}: an FP8 tensor this converter doesn't know")
                put(key, t)
        flush()
    handles.clear()

    n = len(names)
    final = {}
    for j, name in enumerate(names, 1):
        os.replace(work / name, work / f"model-{j:05d}-of-{n:05d}.safetensors")
    for key, name in placed.items():
        final[key] = name.replace(".safetensors", f"-of-{n:05d}.safetensors")
    total = sum((work / name).stat().st_size for name in set(final.values()))
    index = {"metadata": {"total_size": total}, "weight_map": final}
    (work / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {
        "quant_method": "drama3-int4", "bits": 4, "group_size": GROUP, "sym": True, "desc_act": False,
        "experts": {"quant_method": "gptq", "bits": 4, "group_size": GROUP, "sym": True},
        "dense": {"quant_method": "gptq_marlin", "bits": 4, "group_size": GROUP, "sym": True, "desc_act": False,
                  "targets": DENSE_TARGETS},
    }  # fmt: skip
    (work / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for f in src.iterdir():
        if not f.is_file() or f.suffix == ".safetensors" or f.name in ("model.safetensors.index.json", "config.json"):
            continue
        try:
            os.link(f, work / f.name)  # the codec (1.9 GB) needn't be copied
        except OSError:
            shutil.copy2(f, work / f.name)
    for st in stats.values():
        st["mean_rel_err"] = st["rel_err_sum"] / max(1, st["tensors"])
    (work / STATS).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    progress(1.0)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(work, dst)
    return stats


def build(src: Path, dst: Path, progress: Callable[[float], None] | None = None) -> None:
    """models.download's build step for this model."""
    convert(src, dst, progress)


if __name__ == "__main__":  # python -m polyvox.engines.fish_drama3.convert FP8_DIR OUT_DIR
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m polyvox.engines.fish_drama3.convert FP8_DIR OUT_DIR")
    print(json.dumps(convert(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
