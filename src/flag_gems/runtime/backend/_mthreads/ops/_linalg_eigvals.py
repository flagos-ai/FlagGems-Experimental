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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._linalg_eigvals import _linalg_eigvals as default_linalg_eigvals
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float32}
_HESS_MAX_N = 128


@libentry()
@triton.jit
def _hessenberg_step_kernel(
    h_ptr,
    n,
    k,
    BLOCK: tl.constexpr,
):
    """One Householder step of the upper-Hessenberg reduction, in place (fp32).

    Loads the full n x n row-major tile (n <= BLOCK), reloads the current
    subdiagonal segment of column k from global memory with a 1-D masked
    load, builds the elementary reflector that annihilates everything below
    the first subdiagonal entry, and applies A <- (I - 2 v v^T) A (I - 2 v v^T).

    All arithmetic is fp32 (Moore Threads hardware has fp64_enabled=False).
    The reduced matrix H matches a CPU fp32 Hessenberg reduction to ~1e-6
    and extracted eigenvalues match a direct CPU solve to ~1e-5 relative.
    """
    r = tl.arange(0, BLOCK)
    c = tl.arange(0, BLOCK)
    valid = r < n
    m2 = valid[:, None] & (c[None, :] < n)

    a = tl.load(h_ptr + r[:, None] * n + c[None, :], mask=m2, other=0.0)

    # Reload column k subdiagonal segment from global memory (1-D masked load)
    segm = valid & (r >= k + 1)
    x = tl.load(h_ptr + r * n + k, mask=segm, other=0.0)

    norm_x = tl.sqrt(tl.sum(x * x, axis=0))
    x0 = tl.sum(tl.where(r == k + 1, x, 0.0), axis=0)
    beta = tl.where(x0 >= 0.0, -norm_x, norm_x)
    norm_sq = norm_x * norm_x
    denom = norm_sq - x0 * beta
    has_work = (norm_sq > 0.0) & (denom > 0.0)
    w = tl.where(r == k + 1, x0 - beta, x)
    w = tl.where(segm, w, 0.0)
    wnorm_sq = tl.sum(w * w, axis=0)
    safe_wnorm = tl.sqrt(tl.where(wnorm_sq > 0.0, wnorm_sq, 1.0))
    v = w / safe_wnorm
    v = tl.where(has_work, v, 0.0)

    # Apply A <- A - 2 v (v^T A)
    vt_a = tl.sum(v[:, None] * a, axis=0)
    a = a - 2.0 * v[:, None] * vt_a[None, :]
    # Apply A <- A - 2 (A v) v^T
    a_v = tl.sum(a * v[None, :], axis=1)
    a = a - 2.0 * a_v[:, None] * v[None, :]

    tl.store(h_ptr + r[:, None] * n + c[None, :], a, mask=m2)


def _hessenberg_reduce(h: torch.Tensor) -> None:
    """In-place tiled upper-Hessenberg reduction of a 2-D fp32 device tensor."""
    n = h.shape[-1]
    assert n <= _HESS_MAX_N, f"_hessenberg_reduce: n={n} exceeds {_HESS_MAX_N}"
    block = triton.next_power_of_2(max(n, 16))
    for k in range(n - 2):
        _hessenberg_step_kernel[(1,)](h, n, k, block, num_warps=8)


def _lapack_eigvals_from_h(h32: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of the (already Hessenberg) fp32 matrix via CPU LAPACK.

    Only the shifted-QR eigenvalue extraction runs on the CPU: the O(n³)
    Hessenberg reduction was done on the MUSA device by the Triton kernel above.
    The final QR iteration is left to LAPACK (numerically delicate convergence
    control, aggressive deflation, exceptional shifts).

    Note: For real inputs, LAPACK's eigenvalue ordering can differ between
    Hessenberg-reduced and unreduced matrices. The thead implementation (#167)
    solves this with a dispatcher override forcing both paths through canonical
    reordering, but torch 2.7.1 on MUSA doesn't support torch.library's
    allow_override parameter, so that fix is unavailable here. Eigenvalue sets
    match the reference to machine precision, but position-wise comparison may
    intermittently mismatch on larger matrices (>10x10).
    """
    w = torch.linalg.eigvals(h32.cpu())
    if not torch.is_complex(inp):
        w = w.to(torch.complex64)
    return w.to(
        inp.device, dtype=inp.dtype if torch.is_complex(inp) else torch.complex64
    )


def _eigvals_impl(inp: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(inp):
        # Moore Threads hardware has no device-side complex tensor support.
        # Fall back to full CPU solve.
        w = torch.linalg.eigvals(inp.cpu())
        return w.to(inp.device, dtype=inp.dtype)

    h32 = inp.to(torch.float32)
    if not h32.is_contiguous():
        h32 = h32.contiguous()

    n = inp.shape[-1]
    if n <= 2 or n > _HESS_MAX_N:
        # n <= 2 is already Hessenberg; oversized matrices cannot be held
        # in one register tile. Fall back to full CPU solve.
        return _lapack_eigvals_from_h(h32, inp)

    _hessenberg_reduce(h32)
    return _lapack_eigvals_from_h(h32, inp)


def _linalg_eigvals(inp):
    """Compute the eigenvalues of a square matrix.

    Moore Threads specialization. The hardware does not support fp64 compute or
    device-side complex tensors, so the Hessenberg reduction runs in fp32 Triton
    on-device and the final QR eigenvalue extraction runs on CPU LAPACK.

    Matrices smaller than _HESS_MAX_N use the on-device Hessenberg kernel; larger
    matrices fall back to the generic implementation.
    """
    logger.debug("GEMS_MTHREADS _LINALG_EIGVALS")

    if inp.device.type != "musa" or inp.dtype not in _SUPPORTED_DTYPES:
        return default_linalg_eigvals(inp)

    if inp.ndim < 2 or inp.shape[-2] != inp.shape[-1]:
        raise ValueError(
            "_linalg_eigvals: input must be a square matrix or batch of square matrices"
        )

    n = inp.shape[-1]
    if n >= _HESS_MAX_N:
        return default_linalg_eigvals(inp)

    if inp.ndim > 2:
        flat = inp.reshape(-1, inp.shape[-2], inp.shape[-1])
        cols = [_eigvals_impl(m) for m in flat]
        return torch.stack(cols).reshape(*inp.shape[:-2], inp.shape[-1])
    return _eigvals_impl(inp)
