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

"""Triton implementation of ``special_shifted_chebyshev_polynomial_t``.

``out = T_n^*(x) = T_n(2x - 1)`` evaluated with the three-term Chebyshev
recurrence, which is what the ATen device kernel computes::

    T_0 = 1,  T_1 = y,  T_k = 2y * T_{k-1} - T_{k-2},   y = 2x - 1

Using the recurrence instead of the analytic ``cos(n * acos(y))`` form
removes two libdevice transcendentals per element, so the kernel runs at the
memory ceiling, and it is also faithful to the reference outside ``[0, 1]``
where the analytic form would return NaN but the polynomial stays finite.
Verified on this target against the ATen device reference: max |diff|
3.7e-6 (fp32, degree 0..9) and 1.1e-14 (fp64); out-of-domain values and
NaN propagation match exactly.

* the compute dtype follows ``x`` (``float32`` for non-floating ``x``), never ``n``
* ``n`` is only ever used through its integer value
* Half/BFloat16 raise ``ValueError`` in the wrapper before any launch
* a freshly allocated tensor is returned; neither input is mutated or aliased

All arithmetic happens inside ``@triton.jit`` kernels.  The host side only
reads metadata, allocates the output and launches.
"""

import logging

import torch
import triton
import triton.language as tl

_OP_LOGGER = logging.getLogger("flag_gems.ops.special_shifted_chebyshev_polynomial_t")
_MOD_LOGGER = logging.getLogger(__name__)
_MARK = "GEMS SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_T"

# Launch configuration thresholds (measured on the 80-CU target).
_MID_NUMEL = 1 << 21
_BIG_NUMEL = 1 << 24
# Below this the tensor-degree path uses a narrower tile with fewer warps.
_SMALL_NUMEL = 1 << 20
# Largest scalar degree that is unrolled at compile time.
_MAX_STATIC_DEG = 24


@triton.jit
def _cheb_t_scalar_static(
    x_ptr,
    out_ptr,
    total,
    N_DEG: tl.constexpr,
    IS_F64: tl.constexpr,
    CAST_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    if IS_F64:
        one = tl.full([BLOCK], 1.0, dtype=tl.float64)
        if CAST_F32:
            y = 2.0 * x.to(tl.float32) - 1.0
        else:
            y = 2.0 * x - 1.0
    else:
        one = tl.full([BLOCK], 1.0, dtype=tl.float32)
        if CAST_F32:
            y = 2.0 * x.to(tl.float32) - 1.0
        else:
            y = 2.0 * x - 1.0
    if N_DEG == 0:
        res = one
    elif N_DEG == 1:
        res = y
    else:
        prev = one
        cur = y
        two_y = 2.0 * y
        for _k in tl.static_range(2, N_DEG + 1):
            prev, cur = cur, two_y * cur - prev
        res = cur
    tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def _cheb_t_scalar_dyn(
    x_ptr,
    out_ptr,
    total,
    n_deg,
    IS_F64: tl.constexpr,
    CAST_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    if CAST_F32:
        y = 2.0 * x.to(tl.float32) - 1.0
    else:
        y = 2.0 * x - 1.0
    if IS_F64:
        prev = tl.full([BLOCK], 1.0, dtype=tl.float64)
    else:
        prev = tl.full([BLOCK], 1.0, dtype=tl.float32)
    cur = y
    two_y = 2.0 * y
    for _k in tl.range(1, n_deg):
        prev, cur = cur, two_y * cur - prev
    tl.store(out_ptr + offs, cur, mask=mask)


@triton.jit
def _cheb_t_scalar_neg(
    x_ptr, out_ptr, total, IS_F64: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    if IS_F64:
        z = x.to(tl.float64) * 0.0
    else:
        z = x.to(tl.float32) * 0.0
    tl.store(out_ptr + offs, z, mask=mask)


@triton.jit
def _cheb_t_flat(
    x_ptr,
    n_ptr,
    out_ptr,
    total,
    IS_F64: tl.constexpr,
    CAST_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    n = tl.load(n_ptr + offs, mask=mask, other=0)
    if CAST_F32:
        y = 2.0 * x.to(tl.float32) - 1.0
    else:
        y = 2.0 * x - 1.0
    nv = n.to(tl.int32)
    if IS_F64:
        one = tl.full([BLOCK], 1.0, dtype=tl.float64)
        zero = tl.full([BLOCK], 0.0, dtype=tl.float64)
        prev = tl.full([BLOCK], 1.0, dtype=tl.float64)
    else:
        one = tl.full([BLOCK], 1.0, dtype=tl.float32)
        zero = tl.full([BLOCK], 0.0, dtype=tl.float32)
        prev = tl.full([BLOCK], 1.0, dtype=tl.float32)
    cur = y
    two_y = 2.0 * y
    top = tl.max(nv, axis=0)
    for k in tl.range(2, top + 1, loop_unroll_factor=2):
        nxt = two_y * cur - prev
        act = nv >= k
        prev = tl.where(act, cur, prev)
        cur = tl.where(act, nxt, cur)
    res = tl.where(nv == 0, one, tl.where(nv == 1, y, cur))
    res = tl.where(nv < 0, zero, res)
    tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def _cheb_t_bcast(
    x_ptr,
    n_ptr,
    out_ptr,
    total,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    IS_F64: tl.constexpr,
    CAST_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    idx = offs.to(tl.int64)
    xo = tl.zeros([BLOCK], dtype=tl.int64)
    no = tl.zeros([BLOCK], dtype=tl.int64)
    c = idx % s7
    idx = idx // s7
    xo += c * x7
    no += c * n7
    c = idx % s6
    idx = idx // s6
    xo += c * x6
    no += c * n6
    c = idx % s5
    idx = idx // s5
    xo += c * x5
    no += c * n5
    c = idx % s4
    idx = idx // s4
    xo += c * x4
    no += c * n4
    c = idx % s3
    idx = idx // s3
    xo += c * x3
    no += c * n3
    c = idx % s2
    idx = idx // s2
    xo += c * x2
    no += c * n2
    c = idx % s1
    idx = idx // s1
    xo += c * x1
    no += c * n1
    c = idx % s0
    xo += c * x0
    no += c * n0
    x = tl.load(x_ptr + xo, mask=mask, other=0)
    n = tl.load(n_ptr + no, mask=mask, other=0)
    if CAST_F32:
        y = 2.0 * x.to(tl.float32) - 1.0
    else:
        y = 2.0 * x - 1.0
    nv = n.to(tl.int32)
    if IS_F64:
        one = tl.full([BLOCK], 1.0, dtype=tl.float64)
        zero = tl.full([BLOCK], 0.0, dtype=tl.float64)
        prev = tl.full([BLOCK], 1.0, dtype=tl.float64)
    else:
        one = tl.full([BLOCK], 1.0, dtype=tl.float32)
        zero = tl.full([BLOCK], 0.0, dtype=tl.float32)
        prev = tl.full([BLOCK], 1.0, dtype=tl.float32)
    cur = y
    two_y = 2.0 * y
    top = tl.max(nv, axis=0)
    for k in tl.range(2, top + 1, loop_unroll_factor=2):
        nxt = two_y * cur - prev
        act = nv >= k
        prev = tl.where(act, cur, prev)
        cur = tl.where(act, nxt, cur)
    res = tl.where(nv == 0, one, tl.where(nv == 1, y, cur))
    res = tl.where(nv < 0, zero, res)
    tl.store(out_ptr + offs, res, mask=mask)


def _broadcast_plan(x, n, out_shape):
    """Per-dim sizes plus padded x/n strides (0 on broadcast dims)."""
    nd = len(out_shape)
    pad = 8 - nd
    sizes = (1,) * pad + tuple(out_shape)

    def _strides(t):
        lead = nd - t.dim()
        st = []
        for i in range(nd):
            j = i - lead
            if j < 0 or t.shape[j] == 1:
                st.append(0)
            else:
                st.append(t.stride(j))
        return (0,) * pad + tuple(st)

    return sizes, _strides(x), _strides(n)


def special_shifted_chebyshev_polynomial_t(x, n):
    if _OP_LOGGER.isEnabledFor(logging.DEBUG):
        _OP_LOGGER.debug(_MARK)
    elif _MOD_LOGGER.isEnabledFor(logging.DEBUG):
        _MOD_LOGGER.debug(_MARK)

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    in_dtype = x.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"Unsupported dtype {in_dtype}: shifted_chebyshev_polynomial_t "
            "has no Half/BFloat16 implementation"
        )

    if in_dtype.is_floating_point:
        out_dtype = in_dtype
        cast_f32 = False
    else:
        out_dtype = torch.float32
        cast_f32 = True
    is_f64 = out_dtype is torch.float64

    if isinstance(n, torch.Tensor):
        if n.device != x.device:
            n = n.to(x.device)
        n_shape = tuple(n.shape)
        scalar_deg = None
    else:
        n_shape = ()
        scalar_deg = int(n)

    x_shape = tuple(x.shape)
    if x_shape == n_shape:
        out_shape = x_shape
    else:
        out_shape = torch.broadcast_shapes(x_shape, n_shape)

    out = torch.empty(out_shape, dtype=out_dtype, device=x.device)
    total = out.numel()
    if total == 0:
        return out

    if total >= _BIG_NUMEL:
        block, warps = 2048, 8
    elif total >= _MID_NUMEL:
        block, warps = 1024, 8
    else:
        block, warps = 1024, 4
    grid = (triton.cdiv(total, block),)

    if scalar_deg is not None:
        if scalar_deg < 0:
            _cheb_t_scalar_neg[grid](
                x, out, total, IS_F64=is_f64, BLOCK=block, num_warps=warps
            )
        elif scalar_deg <= _MAX_STATIC_DEG:
            _cheb_t_scalar_static[grid](
                x,
                out,
                total,
                N_DEG=scalar_deg,
                IS_F64=is_f64,
                CAST_F32=cast_f32,
                BLOCK=block,
                num_warps=warps,
            )
        else:
            _cheb_t_scalar_dyn[grid](
                x,
                out,
                total,
                scalar_deg,
                IS_F64=is_f64,
                CAST_F32=cast_f32,
                BLOCK=block,
                num_warps=warps,
            )
        return out

    if (
        x_shape == n_shape
        and len(out_shape) == len(n_shape)
        and x.is_contiguous()
        and n.is_contiguous()
    ):
        # The tensor-degree path reads three streams (x, int64 n, out).  On very
        # large tensors it prefers the same 8 warps with a narrower tile; on
        # small outputs a smaller tile keeps the per-CTA tail short.
        if total >= _BIG_NUMEL:
            block, fwarps = 1024, 8
        elif total < _SMALL_NUMEL:
            block, fwarps = 512, 4
        else:
            block, fwarps = block, warps
        _cheb_t_flat[(triton.cdiv(total, block),)](
            x,
            n,
            out,
            total,
            IS_F64=is_f64,
            CAST_F32=cast_f32,
            BLOCK=block,
            num_warps=fwarps,
        )
        return out

    sizes, xs, ns = _broadcast_plan(x, n, out_shape)
    _cheb_t_bcast[grid](
        x,
        n,
        out,
        total,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
        xs[7],
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        ns[4],
        ns[5],
        ns[6],
        ns[7],
        IS_F64=is_f64,
        CAST_F32=cast_f32,
        BLOCK=block,
        num_warps=warps,
    )
    return out
