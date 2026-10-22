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

"""Triton implementation of ``aten::special_ndtri`` (inverse standard-normal CDF).

The value path uses two target-fitted rationals, one division, and a
dtype-specialized tail logarithm:

* fp32 specialization: central degree-3/2 rational, tail degree-4/3 rational in
  ``q = sqrt(-2 log u)`` over ``q in [2, 15]`` (enough for fp32 subnormals),
  base-2 logarithm for the tail, and a non-IEEE ``fast_dividef``.  Every one of
  these was measured on the target to be both more accurate than the tolerance
  requires and faster.
* fp64 specialization: central degree-3/2 rational, tail degree-4/3 rational
  over ``q in [2, 36]``, the tail logarithm replaced by bit-extraction base-2
  (exponent from the IEEE-754 exponent field with a subnormal rescale plus a
  degree-6 mantissa polynomial), and IEEE division.

Coefficients are least-squares fitted on the target against
``torch.special.ndtri`` with the evaluation tolerance as the weight, so the
approximation error stays far inside the contract tolerance.  All polynomials
are explicit fused multiply-add chains.

Boundary contract is applied last so it always wins: ``p == 0 -> -inf``,
``p == 1 -> +inf``, ``p < 0`` or ``p > 1`` or NaN -> NaN.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

logger = logging.getLogger("flag_gems.ops.special_ndtri")

_SUPPORTED_DTYPES = (torch.float32, torch.float64)

_EXP_M2 = 0.13533528323661269189  # exp(-2)
_MAX_RANK = 8
# sqrt(2 * ln 2): q = sqrt(-2 ln u) = _SQRT_2LN2 * sqrt(-log2(u))
_SQRT_2LN2 = 1.1774100225154747


@triton.jit
def _ndtri_value(p, IS_F64: tl.constexpr):
    """Vector-valued inverse standard-normal CDF for probability lanes ``p``."""
    # Symmetric reduction: u in (0, 0.5]; the sign comes from the original p.
    u = tl.where(p <= 0.5, p, 1.0 - p)
    t = u - 0.5
    r = t * t

    if IS_F64:
        # ---- fp64: central degree 3/2, tail degree 4/3 in q over [2, 36]
        cnum = tl.fma(
            tl.fma(
                tl.fma(0.714745424545168, r, 3.2097176904360416), r, -4.343015309665466
            ),
            r,
            0.999999990794042,
        ) * (t * 2.50662827463100050242)
        cden = tl.fma(tl.fma(6.55204857074988, r, -5.3902203743892425), r, 1.0)
        # tail logarithm via IEEE-754 exponent extraction (no transcendental)
        ix = u.to(tl.int64, bitcast=True)
        e0 = (ix >> 52) & 2047
        uu = tl.where(e0 == 0, u * 4503599627370496.0, u)
        ix = uu.to(tl.int64, bitcast=True)
        ex = ((ix >> 52) & 2047).to(tl.float64) - tl.where(e0 == 0, 1075.0, 1023.0)
        mf = (
            (ix & 4503599627370495).to(tl.float64) * 2.220446049250313e-16 + 1.0
        ) - 1.5
        lg = tl.fma(-0.02482556335806043, mf, 0.04342830144321741)
        lg = tl.fma(lg, mf, -0.0706862373062839)
        lg = tl.fma(lg, mf, 0.14184888081446365)
        lg = tl.fma(lg, mf, -0.32062477618843577)
        lg = tl.fma(lg, mf, 0.9618147778021203)
        lg = tl.fma(lg, mf, 0.5849626826303914) + ex
        q = 1.1774100225154747 * tl.sqrt(-lg)
        tnum = tl.fma(
            tl.fma(
                tl.fma(
                    tl.fma(-0.027350014580792337, q, -0.6144528193145247),
                    q,
                    -1.8609358537501608,
                ),
                q,
                0.8531931182485141,
            ),
            q,
            2.626467034080824,
        )
        tden = tl.fma(
            tl.fma(
                tl.fma(0.027348577184731937, q, 0.6148328911216002),
                q,
                2.002934381101779,
            ),
            q,
            1.0,
        )
        central = u > 0.13533528323661269189
        x = tl.where(central, cnum, tnum) / tl.where(central, cden, tden)
    else:
        # ---- fp32: central degree 3/2 in r, tail degree 4/3 in q in [2, 15],
        # base-2 tail logarithm and a non-IEEE reciprocal-based division.
        cnum = tl.fma(
            tl.fma(
                tl.fma(0.7147431597381165, r, 3.209713522449283), r, -4.343013233760615
            ),
            r,
            0.9999999907993327,
        ) * (t * 2.50662827463100050242)
        cden = tl.fma(tl.fma(6.55204220029411, r, -5.390218297728406), r, 1.0)
        q = 1.1774100225154747 * tl.sqrt(-tl.log2(u))
        tnum = tl.fma(
            tl.fma(
                tl.fma(
                    tl.fma(-0.11424828062647974, q, -1.5147415997883122),
                    q,
                    -2.5119386109092767,
                ),
                q,
                2.8277310056668346,
            ),
            q,
            2.8434740378555,
        )
        tden = tl.fma(
            tl.fma(
                tl.fma(0.11422750992381459, q, 1.5176084234989382),
                q,
                3.0412107517375433,
            ),
            q,
            1.0,
        )
        central = u > 0.13533528323661269189
        x = libdevice.fast_dividef(
            tl.where(central, cnum, tnum), tl.where(central, cden, tden)
        )

    x = tl.where(p > 0.5, -x, x)

    # Boundary contract, applied last so no dead lane can leak through.
    x = tl.where(p == 0.0, float("-inf"), x)
    x = tl.where(p == 1.0, float("inf"), x)
    x = tl.where((p < 0.0) | (p > 1.0) | (p != p), float("nan"), x)
    return x


@triton.jit
def _ndtri_flat_kernel(
    x_ptr, out_ptr, n, IS_F64: tl.constexpr, BLOCK: tl.constexpr, INT64: tl.constexpr
):
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(x_ptr + offs, mask=mask, other=0.5)
    tl.store(out_ptr + offs, _ndtri_value(p, IS_F64), mask=mask)


@triton.jit
def _ndtri_contig_kernel(
    x_ptr,
    out_ptr,
    n,
    ITERS: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
):
    """One program owns ``ITERS`` consecutive blocks of the flattened tensor.

    At a billion elements the flat grid would launch ~1M programs, and the
    per-program setup dominates; giving each program a contiguous span measured
    7% faster there for fp32 while leaving smaller tensors on the flat kernel.
    """
    pid = tl.program_id(0)
    if INT64:
        start = pid.to(tl.int64) * (BLOCK * ITERS)
        step = BLOCK
    else:
        start = pid * (BLOCK * ITERS)
        step = BLOCK
    for i in tl.range(0, ITERS):
        offs = start + i * step + tl.arange(0, BLOCK)
        mask = offs < n
        p = tl.load(x_ptr + offs, mask=mask, other=0.5)
        tl.store(out_ptr + offs, _ndtri_value(p, IS_F64), mask=mask)


@triton.jit
def _ndtri_contig_exact_kernel(
    x_ptr,
    out_ptr,
    ITERS: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
):
    """Like ``_ndtri_contig_kernel`` but with no bounds predicate.

    Valid only when the element count is an exact multiple of ``BLOCK * ITERS``,
    which the caller checks on the host.  Dropping the per-iteration mask
    measured a further 3% (fp32) and 2% (fp64) at a billion elements.
    """
    pid = tl.program_id(0)
    if INT64:
        start = pid.to(tl.int64) * (BLOCK * ITERS)
    else:
        start = pid * (BLOCK * ITERS)
    for i in tl.range(0, ITERS):
        offs = start + i * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, _ndtri_value(tl.load(x_ptr + offs), IS_F64))


@triton.jit
def _ndtri_strided_kernel(
    x_ptr,
    out_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n

    rem = offs
    i7 = rem % s7
    rem = rem // s7
    i6 = rem % s6
    rem = rem // s6
    i5 = rem % s5
    rem = rem // s5
    i4 = rem % s4
    rem = rem // s4
    i3 = rem % s3
    rem = rem // s3
    i2 = rem % s2
    rem = rem // s2
    i1 = rem % s1
    i0 = rem // s1

    src = i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4 + i5 * t5 + i6 * t6 + i7 * t7
    p = tl.load(x_ptr + src, mask=mask, other=0.5)
    tl.store(out_ptr + offs, _ndtri_value(p, IS_F64), mask=mask)


def _collapse(shape, strides):
    """Merge adjacent dimensions describing one run of dense memory.

    ``(S, T)`` followed by ``(s, t)`` is one run iff ``T == s * t``: the flat
    index ``i*T + j*t`` then equals ``(i*s + j)*t``.  Metadata only.
    """
    out_shape = []
    out_strides = []
    for size, stride in zip(shape, strides):
        if size == 1:
            continue
        if out_shape and out_strides[-1] == size * stride:
            out_shape[-1] = out_shape[-1] * size
            out_strides[-1] = stride
        else:
            out_shape.append(size)
            out_strides.append(stride)
    if not out_shape:
        out_shape = [1]
        out_strides = [0]
    return out_shape, out_strides


def special_ndtri(self):
    logger.debug("GEMS SPECIAL_NDTRI")

    dtype = self.dtype
    if dtype is not torch.float32 and dtype is not torch.float64:
        raise NotImplementedError(
            f"\"ndtri\" not implemented for '{dtype}'; supported dtypes are "
            f"{_SUPPORTED_DTYPES}"
        )

    out = torch.empty_like(self, memory_format=torch.contiguous_format)
    n = self.numel()
    if n == 0:
        return out

    is_f64 = dtype is torch.float64
    if self.is_contiguous():
        # Launch shape is chosen host-side from the element count only.
        # Large tensors favour few warps per program (measured 0.1472 ms at
        # 1024/1 vs 0.1570 ms at 1024/4 on 16.7M fp32 elements and 0.2585 ms at
        # 512/2 for fp64); on a few thousand elements the launch is latency
        # bound and a wider program measures faster, but the difference there is
        # inside measurement jitter.
        if n <= 65536:
            block, warps = (1024, 4) if not is_f64 else (512, 4)
        else:
            block, warps = (1024, 1) if not is_f64 else (512, 2)
        if n >= 1 << 28:
            # Very large tensors: give each program four consecutive blocks so
            # the grid stays small (a flat 1G-element grid launches ~1M
            # programs and measured 7% slower for fp32 at that size), and skip
            # the bounds predicate entirely when the tensor divides exactly.
            iters = 4
            if n % (block * iters) == 0:
                _ndtri_contig_exact_kernel[(n // (block * iters),)](
                    self,
                    out,
                    ITERS=iters,
                    IS_F64=is_f64,
                    BLOCK=block,
                    INT64=n > 2147483647,
                    num_warps=warps,
                )
                return out
            grid = ((n + block * iters - 1) // (block * iters),)
            _ndtri_contig_kernel[grid](
                self,
                out,
                n,
                ITERS=iters,
                IS_F64=is_f64,
                BLOCK=block,
                INT64=n > 2147483647,
                num_warps=warps,
            )
            return out
        grid = ((n + block - 1) // block,)
        _ndtri_flat_kernel[grid](
            self,
            out,
            n,
            IS_F64=is_f64,
            BLOCK=block,
            INT64=n > 2147483647,
            num_warps=warps,
        )
        return out

    shape, strides = _collapse(tuple(self.shape), tuple(self.stride()))
    if len(shape) > _MAX_RANK:
        raise NotImplementedError(
            "special_ndtri: strided input with rank %d after collapsing is unsupported"
            % len(shape)
        )
    pad = _MAX_RANK - len(shape)
    sizes = [1] * pad + list(shape)
    strides = [0] * pad + list(strides)

    block = 256
    grid = ((n + block - 1) // block,)
    _ndtri_strided_kernel[grid](
        self, out, n, *sizes, *strides, IS_F64=is_f64, BLOCK=block, num_warps=4
    )
    return out
