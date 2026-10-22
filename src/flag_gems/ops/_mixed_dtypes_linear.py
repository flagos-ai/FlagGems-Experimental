# Copyright 2026, The FlagOS Contributors.
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
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# activation string -> integer code consumed by the kernel
_ACT_MAP = {"none": 0, None: 0, "relu": 1, "silu": 2}

# The device is treated as 80 SMs wide, and a launch wants at least two full
# waves of CTAs behind the dequant -> LDS -> MMA dependency chain.
# The device is 80 SMs wide and two CTAs are meant to be resident per SM, so a
# launch wants at least two full waves (160 CTAs) behind the dequant -> LDS ->
# MMA dependency chain.
_CTA_TARGET = 320
_MAX_SPLIT_K = 4


@triton.jit
def _mixed_dtypes_linear_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    partial_ptr,
    M,
    N,
    K,
    ncols,
    stride_xm,
    stride_wn,
    stride_om,
    SPLIT_K: tl.constexpr,
    INT4: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,  # 0: none, 1: relu, 2: silu
    EPI_SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BM), BM)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # K-range owned by this program.  With SPLIT_K == 1 the whole K is inside a
    # single program and the results are final; with SPLIT_K > 1 each program
    # produces a raw fp32 partial that a second kernel reduces in a fixed order.
    if SPLIT_K > 1:
        sk = tl.program_id(1)
        k_per = tl.cdiv(K, SPLIT_K)
        k_start = sk * k_per
        k_end = tl.minimum(K, k_start + k_per)
    else:
        sk = 0
        k_start = 0
        k_end = K

    offs_k = k_start + tl.arange(0, BK)
    x_ptrs = x_ptr + offs_am[:, None] * stride_xm + offs_k[None, :]

    if INT4:
        # Two 4-bit weights are packed per byte: low nibble -> even output
        # column, high nibble -> odd output column.  Interleaving the two
        # nibble planes inside the k-loop forces a [BK, BN] permutation
        # (through shared memory) on every iteration, so instead each plane is
        # dotted separately against the same activation tile and the two
        # accumulators are interleaved ONCE in the epilogue.
        offs_nb = pid_n * (BN // 2) + tl.arange(0, BN // 2)
        nb_mask = offs_nb < ncols
        w_ptrs = w_ptr + offs_k[:, None] * stride_wn + offs_nb[None, :]
        w_mask_n = nb_mask[None, :]
        acc_lo = tl.zeros((BM, BN // 2), dtype=tl.float32)
        acc_hi = tl.zeros((BM, BN // 2), dtype=tl.float32)
        for _ in range(k_start, k_end, BK):
            k_mask = offs_k < k_end
            a = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w_mask = (offs_k[:, None] < k_end) & w_mask_n
            wb = tl.load(w_ptrs, mask=w_mask, other=0)
            wb32 = wb.to(tl.int32)
            # FasterTransformer biased int4 converter: subtract 8.
            lo = (wb32 & 0xF) - 8
            hi = ((wb32 >> 4) & 0xF) - 8
            if EPI_SCALE:
                acc_lo = tl.dot(a, lo.to(a.dtype), acc_lo)
                acc_hi = tl.dot(a, hi.to(a.dtype), acc_hi)
            else:
                # bf16 reference parity: round bf16((nibble - 8) * scale) per
                # weight element before the matmul, for BOTH nibble planes.
                s_even = tl.load(scale_ptr + 2 * offs_nb, mask=nb_mask, other=0.0)
                s_odd = tl.load(scale_ptr + 2 * offs_nb + 1, mask=nb_mask, other=0.0)
                wf_lo = (lo.to(tl.float32) * s_even.to(tl.float32)[None, :]).to(a.dtype)
                wf_hi = (hi.to(tl.float32) * s_odd.to(tl.float32)[None, :]).to(a.dtype)
                acc_lo = tl.dot(a, wf_lo, acc_lo)
                acc_hi = tl.dot(a, wf_hi, acc_hi)
            x_ptrs += BK
            w_ptrs += BK * stride_wn
            offs_k += BK
        acc = tl.interleave(acc_lo, acc_hi)  # [BM, BN], once per program
    else:
        w_ptrs = w_ptr + offs_k[:, None] * stride_wn + offs_n[None, :]
        w_mask_n = n_mask[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(k_start, k_end, BK):
            k_mask = offs_k < k_end
            a = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w_mask = (offs_k[:, None] < k_end) & w_mask_n
            wb = tl.load(w_ptrs, mask=w_mask, other=0)
            wb32 = wb.to(tl.int32)
            # FasterTransformer biased int8 converter: subtract 128.
            wi = wb32 - 128
            if EPI_SCALE:
                wf = wi.to(a.dtype)
            else:
                s = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
                wf = (wi.to(tl.float32) * s.to(tl.float32)[None, :]).to(a.dtype)
            acc = tl.dot(a, wf, acc)
            x_ptrs += BK
            w_ptrs += BK * stride_wn
            offs_k += BK

    if SPLIT_K > 1:
        # Raw partial: the per-column scale (fp16 path) must multiply the
        # FULL-K sum, so it is applied by the reduction kernel, not here.
        p_ptrs = (
            partial_ptr + sk * M * N + offs_m[:, None] * stride_om + offs_n[None, :]
        )
        tl.store(
            p_ptrs,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
        return

    if EPI_SCALE:
        s_epi = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc * s_epi.to(tl.float32)[None, :]

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + b.to(tl.float32)[None, :]
    if ACT == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:
        acc = acc * tl.sigmoid(acc)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _split_k_reduce_kernel(
    partial_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
    EPI_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for sk in tl.static_range(SPLIT_K):
        acc += tl.load(partial_ptr + sk * M * N + offs, mask=mask, other=0.0)

    offs_n = offs % N
    if EPI_SCALE:
        s = tl.load(scale_ptr + offs_n, mask=mask, other=0.0)
        acc = acc * s.to(tl.float32)
    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_n, mask=mask, other=0.0)
        acc = acc + b.to(tl.float32)
    if ACT == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:
        acc = acc * tl.sigmoid(acc)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _select_config(M, N, int4):
    """Mode-aware tile selection.

    int8: the kernel is latency-bound on the dequant -> LDS -> MMA chain, so a
    64-column N-tile is the right default (it doubles the CTA count on the
    mid-N shapes, 86 -> 172, and removes the 1.075-wave tail on 80 SMs).  The
    exception is the wide-M / wide-N corner: with BM == 64 and N = 11008 each
    CTA re-reads the whole 512 KB activation block, so the activation stream
    (88 MB) dwarfs the 45 MB weight stream.  Widening the N-tile to 256 cuts
    the activation re-reads by 4x and is measurably faster there.

    packed int4: the tile's N extent is split across two independent nibble
    dots, so halving BN to 64 turns each MMA into an N=32 op and doubles the
    per-output-tile weight loads, scale loads and dequant chains; measurement
    shows 128 is the right tile there (2.5x better than 64 on the int4 shapes).
    """
    if M <= 16:
        BM = 16
    elif M <= 32:
        BM = 32
    else:
        BM = 64
    if int4:
        BN = 128
    elif BM == 64 and N >= 8192:
        BN = 256
    else:
        BN = 64
    return BM, BN


def _select_split_k(M, N, K, ncols, BM, BN, int4):
    """Split K only where it cannot hurt.

    Split-K pays because the narrow-M int8 shapes are latency-bound on the
    dequant -> LDS -> MMA chain and a 64-CTA launch leaves most of an 80-SM
    device idle (round-4 profile: VALUBusy ~20 %).  It is refused for the
    packed-int4 shapes (each nibble-plane dot already splits the tile, and the
    measured BN=64 experiment showed that regime punishes narrow tiles), for the
    wide-N int8 exception (BN == 256 already removed the activation re-reads),
    and for short-K shapes, so that the small correctness cases keep exactly the
    accumulation order a single program produces.
    """
    if int4 or BN > 64 or K < 1024:
        return 1
    num_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    want = (_CTA_TARGET + num_tiles - 1) // num_tiles
    # Keep the fp32 partial write+read (8 bytes per output and split) below the
    # size of the weight stream it is meant to overlap.
    weight_bytes = K * ncols
    split = 1
    while split < min(want, _MAX_SPLIT_K):
        nxt = split * 2
        if 8 * M * N * nxt > weight_bytes:
            break
        split = nxt
    while split > 1 and K % split != 0:
        split //= 2
    return split


def _mixed_dtypes_linear(input, weight, scale, *, bias=None, activation=None):
    logger.debug("GEMS _MIXED_DTYPES_LINEAR")
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input must be float16 or bfloat16, got {input.dtype}"
    assert weight.dtype == torch.uint8, f"weight must be uint8, got {weight.dtype}"
    assert (
        scale.dtype == input.dtype
    ), f"scale dtype {scale.dtype} must match input dtype {input.dtype}"
    act = activation if activation is not None else "none"
    assert act in _ACT_MAP, f"unsupported activation: {activation}"

    orig = input.shape
    K = orig[-1]
    M = input.numel() // K
    N = scale.shape[0]
    ncols = weight.shape[1]
    # int8 mode: one byte per output column; int4-packed mode: two per byte.
    assert ncols == N or 2 * ncols == N, (
        f"weight columns {ncols} must equal scale size {N} (int8) "
        f"or half of it (packed int4)"
    )
    int4 = ncols != N

    out = torch.empty((*orig[:-1], N), dtype=input.dtype, device=input.device)

    stride_xm = input.stride(-2) if input.dim() >= 2 else K
    stride_wn = weight.stride(0)
    stride_om = out.stride(-2) if out.dim() >= 2 else N

    epi_scale = input.dtype == torch.float16
    BM, BN = _select_config(M, N, int4)
    BK = 64
    GROUP_M = 8
    # Two pipeline stages: the kernel is latency-bound on the dequant -> LDS ->
    # MMA chain, so the extra shared memory a third stage costs buys nothing
    # and a two-stage ring keeps more CTAs resident per SIMD.
    nstages = 2
    num_warps = 4 if BM <= 32 else 8

    num_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    split_k = _select_split_k(M, N, K, ncols, BM, BN, int4)

    with torch_device_fn.device(input.device):
        if split_k == 1:
            _mixed_dtypes_linear_kernel[(num_tiles,)](
                input,
                weight,
                scale,
                bias if bias is not None else input,
                out,
                out,  # unused partial slot
                M,
                N,
                K,
                ncols,
                stride_xm,
                stride_wn,
                stride_om,
                SPLIT_K=1,
                INT4=int4,
                HAS_BIAS=(bias is not None),
                ACT=_ACT_MAP[act],
                EPI_SCALE=epi_scale,
                BM=BM,
                BN=BN,
                BK=BK,
                GROUP_M=GROUP_M,
                num_warps=num_warps,
                num_stages=nstages,
            )
            return out

        partials = torch.empty(
            (split_k * M * N,), dtype=torch.float32, device=input.device
        )
        _mixed_dtypes_linear_kernel[(num_tiles, split_k)](
            input,
            weight,
            scale,
            bias if bias is not None else input,
            out,
            partials,
            M,
            N,
            K,
            ncols,
            stride_xm,
            stride_wn,
            stride_om,
            SPLIT_K=split_k,
            INT4=int4,
            HAS_BIAS=(bias is not None),
            ACT=_ACT_MAP[act],
            EPI_SCALE=epi_scale,
            BM=BM,
            BN=BN,
            BK=BK,
            GROUP_M=GROUP_M,
            num_warps=num_warps,
            num_stages=nstages,
        )
        REDUCE_BLOCK = 1024
        _split_k_reduce_kernel[(triton.cdiv(M * N, REDUCE_BLOCK),)](
            partials,
            scale,
            bias if bias is not None else input,
            out,
            M,
            N,
            SPLIT_K=split_k,
            HAS_BIAS=(bias is not None),
            ACT=_ACT_MAP[act],
            EPI_SCALE=epi_scale,
            BLOCK=REDUCE_BLOCK,
            num_warps=4,
        )
        return out
