"""Triton kernels for the 4-bit weights (routed experts, and the dense linears: Int4Linear).

``int4_gemv``: rows of x times 4-bit weights picked per row (an expert's slot in a stack), reading
the packed weights directly: int32 [S, K/8, N] (value k at bits 4 * (k % 8) of word k // 8, stored
+ 8) with fp16 scales [S, K/G, N] (G: 128 or another power of two from 8). One program covers
BLOCK_N outputs of one row and walks K a scale group at a time; each packed word is read once.
The routed kernel's experts are always in groups of 128.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _int4_gemv(
    x_ptr,  # [R_in, K] bf16
    x_row_ptr,  # [R] int32: the x row of each output row
    slot_ptr,  # [R] int32: the weight slot of each output row
    qw_ptr,  # [S, K/8, N] int32
    sc_ptr,  # [S, K/G, N] fp16
    out_ptr,  # [SPLIT, R, N] fp32: partial sums over a K range each
    K,
    N,
    BLOCK_N: tl.constexpr,
    WORDS: tl.constexpr,  # packed words per group: G / 8
    SPLIT: tl.constexpr,  # K ranges in programs of their own (small outputs)
):
    row = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    part_id = tl.program_id(2)
    mask_n = offs_n < N
    x_row = tl.load(x_row_ptr + row)
    slot = tl.load(slot_ptr + row).to(tl.int64)
    words = tl.arange(0, WORDS)
    groups = K // (WORDS * 8)
    per = groups // SPLIT
    qw_base = qw_ptr + slot * (K // 8) * N
    sc_base = sc_ptr + slot * groups * N
    x_base = x_ptr + x_row.to(tl.int64) * K
    acc = tl.zeros((WORDS, BLOCK_N), dtype=tl.float32)
    for g in range(part_id * per, part_id * per + per):
        w = tl.load(qw_base + (g * WORDS + words)[:, None] * N + offs_n[None, :], mask=mask_n[None, :], other=0)
        s = tl.load(sc_base + g * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        part = tl.zeros((WORDS, BLOCK_N), dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(x_base + g * WORDS * 8 + words * 8 + j).to(tl.float32)
            q = ((w >> (4 * j)) & 0xF).to(tl.float32) - 8.0
            part += xj[:, None] * q
        acc += part * s[None, :]
    rows = tl.num_programs(0)
    tl.store(out_ptr + (part_id * rows + row) * N + offs_n, tl.sum(acc, axis=0), mask=mask_n)


def _split(programs: int, groups: int, target: int = 128) -> int:
    """K ranges per output block so that at least about ``target`` programs run (a power of two
    dividing ``groups``); wider outputs aren't split (the partial sums cost more than they gain)."""
    split = 1
    while programs * split < target and groups % (split * 2) == 0 and groups // (split * 2) >= 2:
        split *= 2
    return split


@triton.jit
def _int4_dequant(qw_ptr, sc_ptr, out_ptr, K, N, BLOCK_W: tl.constexpr, BLOCK_N: tl.constexpr, WORDS: tl.constexpr):
    """[S, K/8, N] int32 + [S, K/G, N] fp16 -> [S, K, N] bf16; one program per BLOCK_W packed rows."""
    wrow = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)  # over S * K/8
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    w = tl.load(qw_ptr + wrow.to(tl.int64)[:, None] * N + offs_n[None, :], mask=mask_n[None, :], other=0)
    kw = K // 8
    slot = wrow // kw
    group = (wrow % kw) // WORDS
    s = tl.load(
        sc_ptr + (slot * (kw // WORDS) + group).to(tl.int64)[:, None] * N + offs_n[None, :], mask=mask_n[None, :]
    )
    s = s.to(tl.float32)
    base = (slot.to(tl.int64) * K + (wrow % kw).to(tl.int64) * 8)[:, None] * N + offs_n[None, :]
    for j in tl.static_range(8):
        q = ((w >> (4 * j)) & 0xF).to(tl.float32) - 8.0
        tl.store(out_ptr + base + j * N, (q * s).to(tl.bfloat16), mask=mask_n[None, :])


def _words(qweight: torch.Tensor, scales: torch.Tensor) -> int:
    """Packed words per scale group (16 for groups of 128)."""
    words = qweight.shape[-2] // scales.shape[-2]
    assert words * scales.shape[-2] == qweight.shape[-2] and words & (words - 1) == 0, (qweight.shape, scales.shape)
    return words


def int4_dequant(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[S, K/8, N] int32 + [S, K/G, N] fp16 -> W [S, K, N] bf16."""
    s, kw, n = qweight.shape
    k = kw * 8
    out = torch.empty(s, k, n, device=qweight.device, dtype=torch.bfloat16)
    block_w, block_n = 16, 128
    _int4_dequant[(triton.cdiv(s * kw, block_w), triton.cdiv(n, block_n))](
        qweight.contiguous(), scales.contiguous(), out, k, n,
        BLOCK_W=block_w, BLOCK_N=block_n, WORDS=_words(qweight, scales), num_warps=4,
    )  # fmt: skip
    return out


@triton.jit
def _int4_gemv_routed(
    x_ptr,  # [R_in, K] bf16
    x_row_ptr,  # [R] int32
    expert_ptr,  # [R] int32: the expert of each output row
    table_ptr,  # [E] int32: its slot on the GPU, or -1
    qw_ptr,  # [S, K/8, N] int32 (GPU slots)
    sc_ptr,  # [S, K/128, N] fp16
    host_qw_ptr,  # [E, K/8, N] int32 (pinned CPU memory, read over PCIe)
    host_sc_ptr,  # [E, K/128, N] fp16
    out_ptr,  # [SPLIT, R, N] fp32
    K,
    N,
    BLOCK_N: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    part_id = tl.program_id(2)
    per = (K // 128) // SPLIT
    mask_n = offs_n < N
    x_row = tl.load(x_row_ptr + row)
    expert = tl.load(expert_ptr + row)
    slot = tl.load(table_ptr + expert)
    if slot >= 0:
        qw_base = qw_ptr + slot.to(tl.int64) * (K // 8) * N
        sc_base = sc_ptr + slot.to(tl.int64) * (K // 128) * N
    else:
        qw_base = host_qw_ptr + expert.to(tl.int64) * (K // 8) * N
        sc_base = host_sc_ptr + expert.to(tl.int64) * (K // 128) * N
    words = tl.arange(0, 16)
    x_base = x_ptr + x_row.to(tl.int64) * K
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for g in range(part_id * per, part_id * per + per):
        w = tl.load(qw_base + (g * 16 + words)[:, None] * N + offs_n[None, :], mask=mask_n[None, :], other=0)
        s = tl.load(sc_base + g * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        part = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(x_base + g * 128 + words * 8 + j).to(tl.float32)
            q = ((w >> (4 * j)) & 0xF).to(tl.float32) - 8.0
            part += xj[:, None] * q
        acc += part * s[None, :]
    rows = tl.num_programs(0)
    tl.store(out_ptr + (part_id * rows + row) * N + offs_n, tl.sum(acc, axis=0), mask=mask_n)


class HostPointer:
    """A pinned CPU tensor handed to a kernel by address: with unified addressing the GPU reads
    it directly over PCIe (Triton takes anything with ``data_ptr``)."""

    def __init__(self, tensor: torch.Tensor):
        assert tensor.is_pinned(), "only pinned memory is visible to the GPU"
        self.tensor = tensor
        self.dtype = tensor.dtype

    def data_ptr(self) -> int:
        return self.tensor.data_ptr()


def int4_gemv_routed(x, x_rows, experts, table, qweight, scales, host_qweight: HostPointer, host_scales: HostPointer):
    """As int4_gemv, with each row's weights found by expert: its GPU slot (``table``), else
    read from the host stacks. Graph-safe: nothing is decided on the CPU."""
    k = x.shape[1]
    n = qweight.shape[2]
    rows = experts.numel()
    block_n = 64
    blocks = triton.cdiv(n, block_n)
    split = _split(rows * blocks, k // 128)
    out = torch.empty(split, rows, n, device=x.device, dtype=torch.float32)
    _int4_gemv_routed[(rows, blocks, split)](
        x, x_rows, experts, table, qweight, scales, host_qweight, host_scales, out, k, n,
        BLOCK_N=block_n, SPLIT=split, num_warps=4,
    )  # fmt: skip
    return out[0] if split == 1 else out.sum(0)


def int4_gemv(x: torch.Tensor, x_rows: torch.Tensor, slots: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor):
    """out[r] = x[x_rows[r]] @ W[slots[r]] (fp32 [R, N]); W [S, K, N] stored as ``qweight`` / ``scales``."""
    k = x.shape[1]
    n = qweight.shape[2]
    assert qweight.shape[1] * 8 == k, (x.shape, qweight.shape)
    rows = slots.numel()
    words = _words(qweight, scales)
    block_n = 64
    blocks = triton.cdiv(n, block_n)
    split = _split(rows * blocks, k // (words * 8))
    out = torch.empty(split, rows, n, device=x.device, dtype=torch.float32)
    _int4_gemv[(rows, blocks, split)](
        x.contiguous(), x_rows, slots, qweight, scales, out, k, n, BLOCK_N=block_n, WORDS=words, SPLIT=split,
        num_warps=4,
    )  # fmt: skip
    return out[0] if split == 1 else out.sum(0)


INT8_GROUP = 128


@triton.jit
def _int8_gemv(
    x_ptr,  # [K] bf16
    w_ptr,  # [K, N] int8
    sc_ptr,  # [K/G, N] fp16
    out_ptr,  # [SPLIT, N] fp32
    K,
    N,
    BLOCK_N: tl.constexpr,
    G: tl.constexpr,
    SPLIT: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    part_id = tl.program_id(1)
    mask_n = offs_n < N
    per = (K // G) // SPLIT
    rows = tl.arange(0, G)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for g in range(part_id * per, part_id * per + per):
        k = g * G + rows
        w = tl.load(w_ptr + k.to(tl.int64)[:, None] * N + offs_n[None, :], mask=mask_n[None, :], other=0).to(tl.float32)
        x = tl.load(x_ptr + k).to(tl.float32)
        s = tl.load(sc_ptr + g * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(w * x[:, None], axis=0) * s
    tl.store(out_ptr + part_id * N + offs_n, acc, mask=mask_n)


class Int8Linear(torch.nn.Module):
    """A linear layer (no bias) in 8 bits: symmetric, one fp16 scale per 128 weights along the input.
    Made from a weight [N, K]; one token reads the 8-bit weights (Triton), more unpack them first."""

    def __init__(self, weight: torch.Tensor, group: int = INT8_GROUP):
        super().__init__()
        n, k = weight.shape
        w = weight.t().float().reshape(k // group, group, n)  # [groups, G, N]
        scale = (w.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-12)
        self.q = (w / scale).round().clamp(-127, 127).to(torch.int8).reshape(k, n).contiguous()
        self.scales = scale.reshape(k // group, n).half().contiguous()
        self.group, self.in_features, self.out_features = group, k, n

    def weight(self, dtype=torch.bfloat16) -> torch.Tensor:
        """The unpacked weight [N, K]."""
        w = self.q.float().reshape(-1, self.group, self.out_features) * self.scales.float()[:, None]
        return w.reshape(self.in_features, self.out_features).t().to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.in_features)
        if flat.shape[0] != 1 or not x.is_cuda:
            return F.linear(x, self.weight(x.dtype))
        n, block_n = self.out_features, 64
        blocks = triton.cdiv(n, block_n)
        split = _split(blocks, self.in_features // self.group, target=512)  # measured best on the tower's shapes
        out = torch.empty(split, n, device=x.device, dtype=torch.float32)
        _int8_gemv[(blocks, split)](
            flat, self.q, self.scales, out, self.in_features, n, BLOCK_N=block_n, G=self.group, SPLIT=split, num_warps=4
        )
        out = out[0] if split == 1 else out.sum(0)
        return out.to(x.dtype).view(*x.shape[:-1], n)


class Int4Linear(torch.nn.Module):
    """A linear layer (no bias) kept in 4 bits: ``qweight`` int32 [K/8, N], ``scales`` fp16 [K/G, N].
    One token reads the packed weights (int4_gemv); more unpack them first."""

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.qweight, self.scales = qweight[None], scales[None]
        self.in_features, self.out_features = qweight.shape[0] * 8, qweight.shape[1]
        self.zero = torch.zeros(1, dtype=torch.int32, device=qweight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.in_features)
        if flat.shape[0] == 1:
            out = int4_gemv(flat, self.zero, self.zero, self.qweight, self.scales).to(x.dtype)
        else:
            out = flat @ int4_dequant(self.qweight, self.scales)[0].to(x.dtype)
        return out.view(*x.shape[:-1], self.out_features)
