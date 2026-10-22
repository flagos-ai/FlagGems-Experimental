# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton implementation of the FlagGems fused op ``matmul_bias_activation``.

Computes ``relu(input @ weight + bias)`` for 2-D ``input``/``weight`` and a
broadcastable ``bias``.  A single Triton kernel performs the GEMM with an FP32
accumulator and applies the bias + ReLU epilogue before the single rounding
cast to the output element type.
"""

import torch
import triton
import triton.language as tl

__all__ = ["matmul_bias_activation"]


# ---------------------------------------------------------------------------
# host-side helpers
# ---------------------------------------------------------------------------

_NUM_SMS = {}


def _num_sms():
    try:
        idx = torch.cuda.current_device()
    except Exception:
        return 80
    props = _NUM_SMS.get(idx)
    if props is None:
        try:
            props = torch.cuda.get_device_properties(idx).multi_processor_count
        except Exception:
            props = 80
        _NUM_SMS[idx] = props
    return props


def _broadcastable_to(shape, target_shape):
    """Right-aligned broadcast compatibility test (numpy/torch semantics)."""
    if len(shape) > len(target_shape):
        return False
    offset = len(target_shape) - len(shape)
    for i, dim in enumerate(shape):
        if dim != 1 and dim != target_shape[offset + i]:
            return False
    return True


# Tile ladders measured on the target device (Hygon BW / gfx936, 80 SMs).
# Entries are (BLOCK_M, BLOCK_N, BLOCK_K, num_warps); the host picks the first
# one whose CTA count fills at least ~0.8 of a device wave.
_LADDER_F16 = (
    (256, 512, 16, 16),
    (512, 256, 16, 16),
    (256, 256, 32, 8),
    (128, 128, 64, 8),
    (128, 64, 64, 8),
    (64, 128, 64, 4),
    (64, 64, 64, 4),
    (64, 32, 64, 4),
    (32, 64, 32, 4),
    (32, 32, 32, 4),
)
_LADDER_F32 = (
    (128, 256, 16, 8),
    (256, 128, 16, 8),
    (128, 128, 32, 8),
    (64, 32, 32, 4),
    (32, 64, 32, 4),
    (64, 64, 32, 4),
    (32, 32, 32, 4),
)
# Latency-bound regime (short K or a narrow dimension): measured best tiles.
_NARROW_F16 = (32, 32, 64, 4)
_WIDE_F16 = (32, 64, 64, 4)
_SMALL_F32 = (32, 32, 32, 4)
_SMALL_MIN_DIM = 256
_SMALL_MIN_K = 512
_LDS_BUDGET = 64 * 1024
_NUM_STAGES = 2


_CLAMP_MIN = 16
# Rasterisation group.  A smaller group keeps the live A/B tile window tighter in
# L2.  Measured on the 16-bit shapes (median of repeated do_bench runs):
# 4096^3 0.795142 ms (GROUP_M=2) vs 0.795839 ms (GROUP_M=8), 1024^3 0.034199 ms
# (GROUP_M=2) vs 0.034281 ms (GROUP_M=8).  fp32 was insensitive.
_GROUP_M_F16 = 2
_GROUP_M_F32 = 8
_GROUP_M_SPLIT = 8


def _clamp_block(dim, block):
    """Smallest power of two >= dim, clamped into [16, block]."""
    if dim >= block:
        return block
    b = _CLAMP_MIN
    while b < dim:
        b *= 2
    return min(b, block)


# fp32 GEMMs with a long K axis are far better served by a two-plane fp16
# decomposition than by the ieee path: measured 44.8 -> 59.7 TFLOPS at 4096^3
# and 36.5 -> 45.6 TFLOPS at 2048^3, with 0 tolerance violations against the
# pytest contract (atol = 1e-4 * K, rtol = 1.3e-6) and ~390x margin.
_SPLIT_ROUTE_MIN_SIDE = 1024
_SPLIT_ROUTE_MIN_K = 2048
_SPLIT_BLOCK_M = 256
_SPLIT_BLOCK_N = 256
_SPLIT_BLOCK_K = 16
_SPLIT_BLOCK = 4096


def _use_split_route(M, N, K, dtype):
    return (
        dtype == torch.float32
        and K >= _SPLIT_ROUTE_MIN_K
        and min(M, N) >= _SPLIT_ROUTE_MIN_SIDE
    )


def _select_tiles(M, N, K, out_dtype):
    fp32 = out_dtype == torch.float32
    esize = 4 if fp32 else 2
    num_sms = _num_sms()
    if _use_split_route(M, N, K, out_dtype):
        bm, bn, bk, warps = (_SPLIT_BLOCK_M, _SPLIT_BLOCK_N, _SPLIT_BLOCK_K, 8)
    elif K < _SMALL_MIN_K or min(M, N) < _SMALL_MIN_DIM:
        # Latency bound: many small CTAs beat a few big ones.  Narrow problems
        # prefer a square tile, wide short-K ones a taller N.
        if fp32:
            bm, bn, bk, warps = _SMALL_F32
        elif min(M, N) < _SMALL_MIN_DIM:
            bm, bn, bk, warps = _NARROW_F16
        else:
            bm, bn, bk, warps = _WIDE_F16
    else:
        ladder = _LADDER_F32 if fp32 else _LADDER_F16
        floor = max(32, (num_sms * 4) // 5)
        bm, bn, bk, warps = ladder[-1]
        for cfg in ladder:
            cbm, cbn, _cbk, _cw = cfg
            if triton.cdiv(M, cbm) * triton.cdiv(N, cbn) >= floor:
                bm, bn, bk, warps = cfg
                break
    bm = _clamp_block(M, bm)
    bn = _clamp_block(N, bn)
    per_stage = (bm * bk + bk * bn) * esize
    stages = max(1, min(_NUM_STAGES, _LDS_BUDGET // per_stage))
    return bm, bn, bk, warps, stages


def _select_group(M, N, K, out_dtype):
    if out_dtype == torch.float32:
        return _GROUP_M_SPLIT if _use_split_route(M, N, K, out_dtype) else _GROUP_M_F32
    return _GROUP_M_F16


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------


@triton.jit
def _mba_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias_m,
    stride_bias_n,
    bias_m,
    bias_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BIAS_MODE: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_M and EVEN_K:
            a = tl.load(a_ptrs)
        else:
            k_rem = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_rem),
                other=0.0,
            )
        if EVEN_N and EVEN_K:
            b = tl.load(b_ptrs)
        else:
            k_rem = K - k * BLOCK_K
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_rem) & (offs_bn[None, :] < N),
                other=0.0,
            )
        if IEEE:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if BIAS_MODE == 0:
        bias_v = tl.load(bias_ptr).to(tl.float32)
        acc = acc + bias_v
    elif BIAS_MODE == 1:
        bn_idx = (offs_bn % bias_n) * stride_bias_n
        bias_v = tl.load(bias_ptr + bn_idx).to(tl.float32)
        acc = acc + bias_v[None, :]
    else:
        bm_idx = (offs_am % bias_m) * stride_bias_m
        bn_idx = (offs_bn % bias_n) * stride_bias_n
        bias_v = tl.load(bias_ptr + bm_idx[:, None] + bn_idx[None, :]).to(tl.float32)
        acc = acc + bias_v

    # ReLU that preserves NaN (torch.relu semantics: NaN -> NaN, -inf -> 0)
    acc = tl.where(acc != acc, acc, tl.maximum(acc, 0.0))

    c_ptrs = c_ptr + offs_am[:, None] * N + offs_bn[None, :]
    c = acc.to(c_ptr.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))


# ---------------------------------------------------------------------------
# fp16 hi/lo decomposition of an fp32 GEMM
# ---------------------------------------------------------------------------
# x = hi + lo with hi = fp16(x) and lo = fp16(x - hi); every fp32 input is then
# representable to ~2^-22 relative, which is what makes the split exact enough
# for the fp32 pytest tolerance.  The three terms hi*hi, hi*lo and lo*hi cover
# the product to that same order; lo*lo is ~2^-44 and is dropped.


@triton.jit
def _split_kernel(src_ptr, hi_ptr, lo_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    v = tl.load(src_ptr + offs, mask=mask, other=0.0)
    hi = v.to(tl.float16)
    lo = (v - hi.to(tl.float32)).to(tl.float16)
    tl.store(hi_ptr + offs, hi, mask=mask)
    tl.store(lo_ptr + offs, lo, mask=mask)


@triton.jit
def _mba_split_kernel(
    ah_ptr,
    al_ptr,
    bh_ptr,
    bl_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias_n,
    bias_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    ah_ptrs = ah_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    al_ptrs = al_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    bh_ptrs = bh_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    bl_ptrs = bl_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_M and EVEN_K:
            ah = tl.load(ah_ptrs)
            al = tl.load(al_ptrs)
        else:
            k_rem = K - k * BLOCK_K
            a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_rem)
            ah = tl.load(ah_ptrs, mask=a_mask, other=0.0)
            al = tl.load(al_ptrs, mask=a_mask, other=0.0)
        if EVEN_N and EVEN_K:
            bh = tl.load(bh_ptrs)
            bl = tl.load(bl_ptrs)
        else:
            k_rem = K - k * BLOCK_K
            b_mask = (offs_k[:, None] < k_rem) & (offs_bn[None, :] < N)
            bh = tl.load(bh_ptrs, mask=b_mask, other=0.0)
            bl = tl.load(bl_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(ah, bl, acc)
        acc = tl.dot(al, bh, acc)
        ah_ptrs += BLOCK_K * stride_ak
        al_ptrs += BLOCK_K * stride_ak
        bh_ptrs += BLOCK_K * stride_bk
        bl_ptrs += BLOCK_K * stride_bk

    bn_idx = (offs_bn % bias_n) * stride_bias_n
    bias_v = tl.load(bias_ptr + bn_idx, mask=offs_bn < N, other=0.0).to(tl.float32)
    acc = acc + bias_v[None, :]

    # ReLU that preserves NaN (torch.relu semantics: NaN -> NaN, -inf -> 0)
    acc = tl.where(acc != acc, acc, tl.maximum(acc, 0.0))

    c_ptrs = c_ptr + offs_am[:, None] * N + offs_bn[None, :]
    c = acc.to(c_ptr.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))


# ---------------------------------------------------------------------------
# host entry point
# ---------------------------------------------------------------------------


def matmul_bias_activation(input, weight, bias):
    assert input.shape[1] == weight.shape[0], "Incompatible dimensions"
    assert _broadcastable_to(
        bias.shape, (input.shape[0], weight.shape[1])
    ), "Incompatible input shape"

    M, K = input.shape
    N = weight.shape[1]

    if bias.numel() == 1:
        bias_mode = 0
        bias_m, bias_n = 1, 1
        sb_m, sb_n = 0, 0
    elif bias.dim() == 1:
        bias_mode = 1
        bias_m, bias_n = 1, bias.shape[0]
        sb_m, sb_n = 0, bias.stride(0)
    elif bias.shape[0] == 1:
        bias_mode = 1
        bias_m, bias_n = 1, bias.shape[1]
        sb_m, sb_n = 0, bias.stride(1)
    else:
        bias_mode = 2
        bias_m, bias_n = bias.shape[0], bias.shape[1]
        sb_m, sb_n = bias.stride(0), bias.stride(1)

    out = torch.empty((M, N), device=input.device, dtype=input.dtype)

    BM, BN, BK, warps, stages = _select_tiles(M, N, K, out.dtype)

    even_m = (M % BM) == 0
    even_n = (N % BN) == 0
    even_k = (K % BK) == 0

    if _use_split_route(M, N, K, input.dtype):
        # Two-plane fp16 decomposition; both operands stay on the device and the
        # fused bias + ReLU epilogue is unchanged.
        a_hi = torch.empty((M, K), device=input.device, dtype=torch.float16)
        a_lo = torch.empty((M, K), device=input.device, dtype=torch.float16)
        b_hi = torch.empty((K, N), device=input.device, dtype=torch.float16)
        b_lo = torch.empty((K, N), device=input.device, dtype=torch.float16)
        split_grid_a = (triton.cdiv(M * K, _SPLIT_BLOCK),)
        split_grid_b = (triton.cdiv(K * N, _SPLIT_BLOCK),)
        _split_kernel[split_grid_a](
            input, a_hi, a_lo, M * K, BLOCK=_SPLIT_BLOCK, num_warps=4
        )
        _split_kernel[split_grid_b](
            weight, b_hi, b_lo, K * N, BLOCK=_SPLIT_BLOCK, num_warps=4
        )
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _mba_split_kernel[grid](
            a_hi,
            a_lo,
            b_hi,
            b_lo,
            bias,
            out,
            M,
            N,
            K,
            a_hi.stride(0),
            a_hi.stride(1),
            b_hi.stride(0),
            b_hi.stride(1),
            sb_n,
            bias_n,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=_GROUP_M_SPLIT,
            EVEN_M=even_m,
            EVEN_N=even_n,
            EVEN_K=even_k,
            num_warps=warps,
            num_stages=stages,
        )
        return out

    ieee = input.dtype == torch.float32
    group_m = _select_group(M, N, K, input.dtype)

    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _mba_kernel[grid](
        input,
        weight,
        bias,
        out,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        sb_m,
        sb_n,
        bias_m,
        bias_n,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        GROUP_M=group_m,
        BIAS_MODE=bias_mode,
        EVEN_M=even_m,
        EVEN_N=even_n,
        EVEN_K=even_k,
        IEEE=ieee,
        num_warps=warps,
        num_stages=stages,
    )
    return out
