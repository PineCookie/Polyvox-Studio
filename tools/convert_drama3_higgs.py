# SPDX-License-Identifier: Apache-2.0
"""Fish Audio Drama 3 in 4 bits: the official FP8 checkpoint -> the HIGGS build.

    python convert_drama3_higgs.py <fishaudio/drama-3 folder> <output folder> [--device cuda|cpu]

Needs only PyTorch and safetensors. About 3 minutes on a GPU (much longer on the CPU), 22 GB of
output. No calibration data. Written into "<output>.build" and renamed when complete.

What it does to the weights (everything else is copied unchanged):
- Routed experts (FP8 with 128 x 128 block scales): dequantized, then each block of 128 input
  weights rotated by a randomized Hadamard matrix and rounded to the 16 levels with the least
  squared error for a standard Gaussian, times one fp16 scale per group of 128, chosen by a search
  for the least error (HIGGS: Malinovskii et al., "Pushing the Limits of Large Language Model
  Quantization"). Stored as ``qweight`` int32 [K/8, N] (value k at bits 4 (k mod 8) of word k / 8,
  an index into the grid) and ``scales`` fp16 [K/128, N]; the rotation's signs per input size K
  as ``quant.hadamard_signs.<K>``.
- Full-attention q / k / v / o and linear-attention in_proj_qkv / in_proj_z / out_proj (bf16):
  symmetric 4-bit integers in groups of 32, stored + 8 as ``qweight`` / ``scales`` with
  ``qzeros`` (all 8) and a sorted ``g_idx``.
The grid and the rotation are described in config.json under ``quantization_config``.

Using the experts: rotate an expert's input the same way (``rotate`` below), then multiply by the
unpacked weights (``dequant_expert``): x W^T = rotate(x) rotate(W)^T.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUP = 128  # the experts' group (and the rotation's block)
DENSE_GROUP = 32
CLIPS = (1.0, 0.81, 20)  # the dense linears' scale search: amax / 7 down to 0.81 x that
# The experts' 16 levels: least squared error for a standard Gaussian (Lloyd-Max).
GRID = (-2.7326, -2.0690, -1.6181, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
        0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326)  # fmt: skip
GRID_SCALES = (1.0, 0.5, 26)  # the experts' scale search: amax / GRID[-1] down to 0.5 x that
SIGNS = "quant.hadamard_signs.{}"
EXPERT = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")
DENSE = re.compile(r"\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj))\.weight$")
DENSE_TARGETS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
]  # fmt: skip
ZERO_POINTS = -2004318072  # 0x88888888: eight 4-bit zero points of 8
SHARD_BYTES = 2_000_000_000
NOTICE_TEXT = (
    "These Fish Audio Materials are licensed under the Fish Audio Community License, Version 1.0. "
    "Copyright © 39 AI, INC. All Rights Reserved.\n\n"
    "Built with Fish Audio.\n\n"
    "This is a Derivative Work of Fish Audio Drama 3 (https://huggingface.co/fishaudio/drama-3). The\n"
    "attribution above applies to the Fish Audio Materials. Changes made to them:\n\n"
    "- The routed experts' weights (FP8 with block scales in the original) were converted to 4 bits:\n"
    "  each block of 128 input weights rotated by a randomized Hadamard matrix, then rounded to 16\n"
    "  levels fitted to a Gaussian, with one fp16 scale per group of 128 (the HIGGS method). The sign\n"
    '  vectors of the rotation are stored as "quant.hadamard_signs.<K>".\n'
    "- The backbone's full-attention q / k / v / o and linear-attention in_proj_qkv / in_proj_z /\n"
    "  out_proj weights (bf16 in the original) were converted to symmetric 4-bit integers with one\n"
    f"  fp16 scale per group of {DENSE_GROUP}.\n"
    '- config.json gained a "quantization_config" describing these formats; quant_stats.json records\n'
    "  the conversion's weight errors.\n"
    "- Everything else (shared experts, routers, embeddings, output head, the codebook decoder, the\n"
    "  audio codec, the tokenizer files and the chat template) is unchanged.\n\n"
    "Converted with convert_drama3_higgs.py (Apache-2.0).\n"
)


def hadamard(n: int = GROUP, device=None) -> torch.Tensor:
    """The orthonormal Hadamard matrix [n, n] (Sylvester), float32."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / math.sqrt(n)).to(device)


def rotation_signs(k: int) -> torch.Tensor:
    """The rotation's ±1 per input position (fixed per K; also stored with the weights)."""
    g = torch.Generator().manual_seed(k)
    return (torch.randint(0, 2, (k,), generator=g) * 2 - 1).to(torch.float32)


def rotate(x: torch.Tensor, signs: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
    """x [..., K] times blockdiag(diag(signs) H) along K, in float32 (returned in x's dtype); the
    same for weights [N, K] and for the inputs they multiply."""
    k = x.shape[-1]
    h = hadamard(GROUP, x.device) if h is None else h
    y = (x.float() * signs.to(x.device)).reshape(*x.shape[:-1], k // GROUP, GROUP) @ h
    return y.reshape(x.shape).to(x.dtype)


def quantize_grid(w: torch.Tensor):
    """float32 [N, K] -> (indices into GRID, uint8 [N, K]; scales float32 [N, K / 128])."""
    n, k = w.shape
    wg = w.view(n, k // GROUP, GROUP)
    levels = torch.tensor(GRID, device=w.device)
    edges = (levels[1:] + levels[:-1]) / 2
    base = wg.abs().amax(-1, keepdim=True).clamp_min(1e-12) / levels.abs().max()
    best_err = torch.full_like(base, float("inf"))
    best_s = base.clone()
    for c in torch.linspace(*GRID_SCALES).tolist():
        s = (base * c).half().float()  # as stored
        err = ((levels[torch.bucketize(wg / s, edges)] * s - wg) ** 2).sum(-1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
    q = torch.bucketize(wg / best_s, edges)
    return q.view(n, k).to(torch.uint8), best_s.squeeze(-1)


def quantize_int4(w: torch.Tensor, group: int):
    """float32 [N, K] -> (q uint8 [N, K] in 0..15 = value + 8, scales float32 [N, K / group])."""
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


def pack(q: torch.Tensor) -> torch.Tensor:
    """q uint8 [N, K] (0..15) -> int32 [K/8, N]."""
    n, k = q.shape
    qt = q.t().contiguous().view(k // 8, 8, n).to(torch.int64)
    shifts = (torch.arange(8, device=q.device, dtype=torch.int64) * 4).view(1, 8, 1)
    word = (qt << shifts).sum(1)
    return (word - (word >= 2**31).to(torch.int64) * 2**32).to(torch.int32)


def unpack(qweight: torch.Tensor) -> torch.Tensor:
    """int32 [..., K/8, N] -> [..., K, N] in 0..15."""
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    return ((qweight.unsqueeze(-2) >> shifts.view(8, 1)) & 0xF).flatten(-3, -2)


def dequant_expert(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """An expert matrix of the build -> its rotated weight^T [K, N] float32: use it with a rotated
    input, rotate(x) @ dequant_expert(...) = x W^T."""
    levels = torch.tensor(GRID, device=qweight.device)
    return levels[unpack(qweight).long()] * scales.float().repeat_interleave(GROUP, dim=-2)


def convert(src: Path, dst: Path, device: str | None = None, report=print) -> dict:
    """Write the 4-bit build of the FP8 checkpoint in ``src`` into ``dst``; returns the error stats."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h = hadamard(GROUP, device)
    signs: dict[int, torch.Tensor] = {}
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

    def record(kind, w, back, src_bytes, int4_bytes):
        err = ((back - w).norm() / w.norm().clamp_min(1e-12)).item()
        st = stats.setdefault(
            kind, {"tensors": 0, "rel_err_sum": 0.0, "rel_err_max": 0.0, "src_bytes": 0, "int4_bytes": 0}
        )
        st["tensors"] += 1
        st["rel_err_sum"] += err
        st["rel_err_max"] = max(st["rel_err_max"], err)
        st["src_bytes"] += src_bytes
        st["int4_bytes"] += int4_bytes

    keys = list(weight_map)
    t0, shown = time.time(), -1
    with torch.inference_mode():
        for i, key in enumerate(keys):
            if (pct := 100 * i // len(keys)) // 5 != shown // 5:
                shown = pct
                report(f"{pct:3d}%  {time.time() - t0:5.0f} s")
            if key.endswith(".weight_scale_inv") and EXPERT.search(key.removesuffix("_scale_inv")):
                continue  # used with its weight
            base = key.removesuffix(".weight")
            if EXPERT.search(key):
                raw, block = get(key), get(key + "_scale_inv")
                if raw.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"{key}: expected FP8 experts, found {raw.dtype}")
                n, k = raw.shape
                scale = block.to(device).float().repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
                w = raw.to(device).float() * scale
                if k not in signs:
                    signs[k] = rotation_signs(k)
                    put(SIGNS.format(k), signs[k])
                w = rotate(w, signs[k], h)
                q, s = quantize_grid(w)
                qw, sc = pack(q), s.t().contiguous().half()
                back = torch.tensor(GRID, device=q.device)[q.long()] * s.float().repeat_interleave(GROUP, 1)
                record("experts", w, back, raw.numel() + block.numel() * 4, qw.numel() * 4 + sc.numel() * 2)
                put(base + ".qweight", qw)
                put(base + ".scales", sc)
            elif DENSE.search(key):
                raw = get(key)
                if raw.dtype != torch.bfloat16 or raw.shape[1] % DENSE_GROUP:
                    raise ValueError(f"{key}: unexpected {raw.dtype} {tuple(raw.shape)}")
                w = raw.to(device).float()
                q, s = quantize_int4(w, DENSE_GROUP)
                n, k = w.shape
                qw, sc = pack(q), s.t().contiguous().half()
                qz = torch.full((k // DENSE_GROUP, math.ceil(n / 8)), ZERO_POINTS, dtype=torch.int32)
                gi = torch.arange(k, dtype=torch.int32) // DENSE_GROUP
                back = (q.float() - 8) * s.float().repeat_interleave(DENSE_GROUP, 1)
                record("dense", w, back, raw.numel() * 2,
                       qw.numel() * 4 + qz.numel() * 4 + sc.numel() * 2 + gi.numel() * 4)  # fmt: skip
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
    for j, name in enumerate(names, 1):
        os.replace(work / name, work / f"model-{j:05d}-of-{n:05d}.safetensors")
    final = {key: name.replace(".safetensors", f"-of-{n:05d}.safetensors") for key, name in placed.items()}
    total = sum((work / name).stat().st_size for name in set(final.values()))
    index = {"metadata": {"total_size": total}, "weight_map": final}
    (work / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {
        "quant_method": "drama3-int4", "bits": 4, "group_size": GROUP, "sym": True, "desc_act": False,
        "experts": {"quant_method": "higgs", "bits": 4, "group_size": GROUP, "grid": list(GRID),
                    "rotation": {"kind": "randomized_hadamard", "block": GROUP, "signs": SIGNS.format("<K>")}},
        "dense": {"quant_method": "gptq_marlin", "bits": 4, "group_size": DENSE_GROUP, "sym": True,
                  "desc_act": False, "targets": DENSE_TARGETS},
    }  # fmt: skip
    (work / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (work / "NOTICE").write_text(NOTICE_TEXT, encoding="utf-8")
    for f in src.iterdir():
        if (
            not f.is_file()
            or f.suffix == ".safetensors"
            or f.name in ("model.safetensors.index.json", "config.json", "NOTICE")
        ):
            continue
        try:
            os.link(f, work / f.name)  # the codec (1.9 GB) needn't be copied
        except OSError:
            shutil.copy2(f, work / f.name)
    for st in stats.values():
        st["mean_rel_err"] = st["rel_err_sum"] / max(1, st["tensors"])
    (work / "quant_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(work, dst)
    report(f"100%  {time.time() - t0:5.0f} s  -> {dst}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src", type=Path, help="the official checkpoint (fishaudio/drama-3)")
    parser.add_argument("dst", type=Path, help="where the 4-bit build goes")
    parser.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    args = parser.parse_args()
    print(json.dumps(convert(args.src, args.dst, args.device), indent=2))


if __name__ == "__main__":
    main()
