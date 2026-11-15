"""Triton kernels for the 4-bit routed experts.

``int4_gemv``: rows of x times 4-bit weights picked per row (an expert's slot in a stack), reading
the packed weights directly: int32 [S, K/8, N] (value k at bits 4 * (k % 8) of word k // 8, stored
+ 8) with fp16 scales [S, K/128, N]. One program covers BLOCK_N outputs of one row and walks K in
groups of 128 (one scale row each); each packed word is read once.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _int4_gemv(
    x_ptr,  # [R_in, K] bf16
    x_row_ptr,  # [R] int32: the x row of each output row
    slot_ptr,  # [R] int32: the weight slot of each output row
    qw_ptr,  # [S, K/8, N] int32
    sc_ptr,  # [S, K/128, N] fp16
    out_ptr,  # [R, N] fp32
    K,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    x_row = tl.load(x_row_ptr + row)
    slot = tl.load(slot_ptr + row).to(tl.int64)
    words = tl.arange(0, 16)  # 16 packed words = one group of 128 weights
    qw_base = qw_ptr + slot * (K // 8) * N
    sc_base = sc_ptr + slot * (K // 128) * N
    x_base = x_ptr + x_row.to(tl.int64) * K
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for g in range(0, K // 128):
        w = tl.load(qw_base + (g * 16 + words)[:, None] * N + offs_n[None, :], mask=mask_n[None, :], other=0)
        s = tl.load(sc_base + g * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        part = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(x_base + g * 128 + words * 8 + j).to(tl.float32)
            q = ((w >> (4 * j)) & 0xF).to(tl.float32) - 8.0
            part += xj[:, None] * q
        acc += part * s[None, :]
    tl.store(out_ptr + row * N + offs_n, tl.sum(acc, axis=0), mask=mask_n)


@triton.jit
def _int4_dequant(qw_ptr, sc_ptr, out_ptr, K, N, BLOCK_W: tl.constexpr, BLOCK_N: tl.constexpr):
    """[S, K/8, N] int32 + [S, K/128, N] fp16 -> [S, K, N] bf16; one program per BLOCK_W packed rows."""
    wrow = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)  # over S * K/8
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    w = tl.load(qw_ptr + wrow.to(tl.int64)[:, None] * N + offs_n[None, :], mask=mask_n[None, :], other=0)
    kw = K // 8
    slot = wrow // kw
    group = (wrow % kw) // 16
    s = tl.load(sc_ptr + (slot * (K // 128) + group).to(tl.int64)[:, None] * N + offs_n[None, :], mask=mask_n[None, :])
    s = s.to(tl.float32)
    base = (slot.to(tl.int64) * K + (wrow % kw).to(tl.int64) * 8)[:, None] * N + offs_n[None, :]
    for j in tl.static_range(8):
        q = ((w >> (4 * j)) & 0xF).to(tl.float32) - 8.0
        tl.store(out_ptr + base + j * N, (q * s).to(tl.bfloat16), mask=mask_n[None, :])


def int4_dequant(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[S, K/8, N] int32 + [S, K/128, N] fp16 -> W [S, K, N] bf16."""
    s, kw, n = qweight.shape
    k = kw * 8
    out = torch.empty(s, k, n, device=qweight.device, dtype=torch.bfloat16)
    block_w, block_n = 16, 128
    _int4_dequant[(triton.cdiv(s * kw, block_w), triton.cdiv(n, block_n))](
        qweight.contiguous(), scales.contiguous(), out, k, n, BLOCK_W=block_w, BLOCK_N=block_n, num_warps=4
    )
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
    out_ptr,  # [R, N] fp32
    K,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
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
    for g in range(0, K // 128):
        w = tl.load(qw_base + (g * 16 + words)[:, None] * N + offs_n[None, :], mask=mask_n[None, :], other=0)
        s = tl.load(sc_base + g * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        part = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(x_base + g * 128 + words * 8 + j).to(tl.float32)
            q = ((w >> (4 * j)) & 0xF).to(tl.float32) - 8.0
            part += xj[:, None] * q
        acc += part * s[None, :]
    tl.store(out_ptr + row * N + offs_n, tl.sum(acc, axis=0), mask=mask_n)


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
    out = torch.empty(rows, n, device=x.device, dtype=torch.float32)
    block_n = 64
    _int4_gemv_routed[(rows, triton.cdiv(n, block_n))](
        x, x_rows, experts, table, qweight, scales, host_qweight, host_scales, out, k, n, BLOCK_N=block_n, num_warps=4
    )
    return out


def int4_gemv(x: torch.Tensor, x_rows: torch.Tensor, slots: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor):
    """out[r] = x[x_rows[r]] @ W[slots[r]] (fp32 [R, N]); W [S, K, N] stored as ``qweight`` / ``scales``."""
    k = x.shape[1]
    n = qweight.shape[2]
    assert qweight.shape[1] * 8 == k and scales.shape[1] * GROUP == k, (x.shape, qweight.shape, scales.shape)
    rows = slots.numel()
    out = torch.empty(rows, n, device=x.device, dtype=torch.float32)
    block_n = 64
    _int4_gemv[(rows, triton.cdiv(n, block_n))](
        x.contiguous(), x_rows, slots, qweight, scales, out, k, n, BLOCK_N=block_n, num_warps=4
    )
    return out
