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

logger = logging.getLogger(__name__)


@triton.jit
def _zero_fill_kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n
    tl.store(ptr + idx, 0.0, mask=mask)


@triton.jit
def _cholesky_step_kernel(
    A_ptr,
    L_ptr,
    n,
    j,
    batch_stride,
    stride_0,
    stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    """One kernel launch per column: diagonal + subdiagonal in one pass."""
    batch_idx = tl.program_id(1)
    row_block = tl.program_id(0)
    base = batch_idx * batch_stride

    # Diagonal: computed by row_block 0
    if row_block == 0:
        acc = 0.0
        for k in range(j):
            val = tl.load(L_ptr + base + j * stride_0 + k * stride_1)
            acc += val * val
        a_jj = tl.load(A_ptr + base + j * stride_0 + j * stride_1)
        diag = tl.sqrt(a_jj - acc)
        tl.store(L_ptr + base + j * stride_0 + j * stride_1, diag)

    # Column: compute L[i,j] for i > j
    row = row_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = (row > j) & (row < n)

    l_jj = tl.load(L_ptr + base + j * stride_0 + j * stride_1)

    dot = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(j):
        l_ik = tl.load(
            L_ptr + base + row * stride_0 + k * stride_1, mask=valid, other=0.0
        )
        l_jk = tl.load(L_ptr + base + j * stride_0 + k * stride_1)
        dot += l_ik * l_jk

    a_ij = tl.load(A_ptr + base + row * stride_0 + j * stride_1, mask=valid, other=0.0)
    l_ij = (a_ij - dot) / l_jj
    tl.store(L_ptr + base + row * stride_0 + j * stride_1, l_ij, mask=valid)


@triton.jit
def _transpose_lower_to_upper(
    L_ptr,
    n,
    batch_stride,
    stride_0,
    stride_1,
):
    """Transpose lower-triangular L to upper-triangular U in-place."""
    batch_idx = tl.program_id(1)
    row = tl.program_id(0)
    base = batch_idx * batch_stride

    for col in range(0, row):
        val = tl.load(L_ptr + base + row * stride_0 + col * stride_1)
        tl.store(L_ptr + base + col * stride_0 + row * stride_1, val)
        tl.store(L_ptr + base + row * stride_0 + col * stride_1, 0.0)


def linalg_cholesky(A, upper=False):
    logger.debug("GEMS_ASCEND LINALG_CHOLESKY_FORWARD")

    shape = A.shape
    ndim = len(shape)
    n = int(shape[-1])

    L = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    numel = int(L.numel())
    _zero_fill_kernel[(triton.cdiv(numel, 1024),)](L, numel, BLOCK_SIZE=1024)

    stride_0 = n
    stride_1 = 1

    batch_size = 1
    for i in range(ndim - 2):
        batch_size *= int(shape[i])
    if batch_size == 0:
        batch_size = 1

    if ndim == 2:
        batch_stride = 0
    else:
        batch_stride = n * n

    BLOCK_SIZE = 256

    # One kernel launch per column: computes diagonal + subdiagonal
    grid = (triton.cdiv(n, BLOCK_SIZE), batch_size)
    for j in range(n):
        _cholesky_step_kernel[grid](
            A,
            L,
            n,
            j,
            batch_stride,
            stride_0,
            stride_1,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # For upper=True, transpose lower to upper in-place
    if upper:
        grid_trans = (n, batch_size)
        _transpose_lower_to_upper[grid_trans](
            L,
            n,
            batch_stride,
            stride_0,
            stride_1,
        )

    return L
