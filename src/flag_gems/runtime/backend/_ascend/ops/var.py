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
def _reduce_sum_1d_kernel(
    x_ptr,
    out_ptr,
    red_size,
    red_stride,
    base_stride_0,
    out_stride_0,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * base_stride_0

    acc = tl.zeros([1], dtype=tl.float32)
    for block_start in tl.range(0, red_size, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < red_size
        vals = tl.load(x_ptr + base + offs * red_stride, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float32))

    acc_sum = tl.sum(acc)
    tl.store(out_ptr + pid * out_stride_0, acc_sum)


@triton.jit
def _reduce_sum_2d_kernel(
    x_ptr,
    out_ptr,
    red_size,
    red_stride,
    base_stride_0,
    base_stride_1,
    out_stride_0,
    out_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    base = pid0 * base_stride_0 + pid1 * base_stride_1

    acc = tl.zeros([1], dtype=tl.float32)
    for block_start in tl.range(0, red_size, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < red_size
        vals = tl.load(x_ptr + base + offs * red_stride, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float32))

    out_idx = pid0 * out_stride_0 + pid1 * out_stride_1
    acc_sum = tl.sum(acc)
    tl.store(out_ptr + out_idx, acc_sum)


@triton.jit
def _reduce_sum_multi_dim_1d_kernel(
    x_ptr,
    out_ptr,
    red_total,
    red_dim1_size,
    red_stride_0,
    red_stride_1,
    base_stride_0,
    out_stride_0,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * base_stride_0

    acc = tl.zeros([1], dtype=tl.float32)
    for block_start in tl.range(0, red_total, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < red_total
        d1 = offs % red_dim1_size
        d0 = offs // red_dim1_size
        elem_off = d0 * red_stride_0 + d1 * red_stride_1
        vals = tl.load(x_ptr + base + elem_off, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float32))

    acc_sum = tl.sum(acc)
    tl.store(out_ptr + pid * out_stride_0, acc_sum)


@triton.jit
def _reduce_sq_diff_1d_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    red_size,
    red_stride,
    base_stride_0,
    out_stride_0,
    correction,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * base_stride_0
    mean_val = tl.load(mean_ptr + pid * out_stride_0)

    acc = tl.zeros([1], dtype=tl.float32)
    for block_start in tl.range(0, red_size, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < red_size
        vals = tl.load(x_ptr + base + offs * red_stride, mask=mask, other=mean_val)
        diff = vals.to(tl.float32) - mean_val
        acc += tl.sum(diff * diff)

    n = tl.maximum(red_size - correction, 1)
    var = acc / n.to(tl.float32)
    var_scalar = tl.sum(var)
    tl.store(out_ptr + pid * out_stride_0, var_scalar)


@triton.jit
def _reduce_sq_diff_2d_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    red_size,
    red_stride,
    base_stride_0,
    base_stride_1,
    out_stride_0,
    out_stride_1,
    correction,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    base = pid0 * base_stride_0 + pid1 * base_stride_1
    out_idx = pid0 * out_stride_0 + pid1 * out_stride_1
    mean_val = tl.load(mean_ptr + out_idx)

    acc = tl.zeros([1], dtype=tl.float32)
    for block_start in tl.range(0, red_size, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < red_size
        vals = tl.load(x_ptr + base + offs * red_stride, mask=mask, other=mean_val)
        diff = vals.to(tl.float32) - mean_val
        acc += tl.sum(diff * diff)

    n = tl.maximum(red_size - correction, 1)
    var = acc / n.to(tl.float32)
    var_scalar = tl.sum(var)
    tl.store(out_ptr + out_idx, var_scalar)


@triton.jit
def _reduce_sq_diff_multi_dim_1d_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    red_total,
    red_dim1_size,
    red_stride_0,
    red_stride_1,
    base_stride_0,
    out_stride_0,
    correction,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * base_stride_0
    mean_val = tl.load(mean_ptr + pid * out_stride_0)

    acc = tl.zeros([1], dtype=tl.float32)
    for block_start in tl.range(0, red_total, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < red_total
        d1 = offs % red_dim1_size
        d0 = offs // red_dim1_size
        elem_off = d0 * red_stride_0 + d1 * red_stride_1
        vals = tl.load(x_ptr + base + elem_off, mask=mask, other=mean_val)
        diff = vals.to(tl.float32) - mean_val
        acc += tl.sum(diff * diff)

    n = tl.maximum(red_total - correction, 1)
    var = acc / n.to(tl.float32)
    var_scalar = tl.sum(var)
    tl.store(out_ptr + pid * out_stride_0, var_scalar)


@triton.jit
def _reduce_sum_sq_1d_kernel(
    x_ptr,
    sum_ptr,
    sq_ptr,
    red_size,
    red_stride,
    base_stride_0,
    out_stride_0,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused sum and sum-of-squares for 1D reduction."""
    pid = tl.program_id(0)
    base = pid * base_stride_0

    acc_sum = tl.zeros([1], dtype=tl.float32)
    acc_sq = tl.zeros([1], dtype=tl.float32)
    for block_start in tl.range(0, red_size, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < red_size
        vals = tl.load(x_ptr + base + offs * red_stride, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        acc_sum += tl.sum(vals_f32)
        acc_sq += tl.sum(vals_f32 * vals_f32)

    idx = pid * out_stride_0
    tl.store(sum_ptr + idx, tl.sum(acc_sum))
    tl.store(sq_ptr + idx, tl.sum(acc_sq))


@triton.jit
def _mean_var_from_sum_sq_1d_kernel(
    sum_ptr,
    sq_ptr,
    out_ptr,
    red_size,
    correction,
    out_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute mean and variance from sum and sum_sq for 1D output."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < out_numel

    s = tl.load(sum_ptr + offs, mask=mask, other=0.0)
    sq = tl.load(sq_ptr + offs, mask=mask, other=0.0)

    mean = s / red_size
    var_pop = sq / red_size - mean * mean
    n_adj = red_size - correction
    if n_adj < 1:
        n_adj = 1
    var = var_pop * red_size / n_adj
    tl.store(out_ptr + offs, var, mask=mask)


@triton.jit
def _div_scalar_kernel(ptr, n, divisor, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    vals = tl.load(ptr + offs, mask=mask, other=0.0)
    tl.store(ptr + offs, vals / divisor, mask=mask)


@triton.jit
def _reduce_sum_2d_contig_kernel(
    x_ptr,
    out_ptr,
    red_size,
    red_stride,
    base_stride_0,
    out_dim1,
    out_stride_0,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce over a single dim for 2D output with contiguous memory access.
    Each block handles one value of output dim 0, iterating over the
    reduction dim with stride-1 loads into output dim 1 tiles."""
    pid = tl.program_id(0)
    base = pid * base_stride_0

    for k_start in range(0, out_dim1, BLOCK_SIZE):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offs < out_dim1
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for j in tl.range(0, red_size, 1):
            vals = tl.load(
                x_ptr + base + j * red_stride + k_offs, mask=k_mask, other=0.0
            )
            acc += vals.to(tl.float32)

        out_offs = pid * out_stride_0 + k_start + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + out_offs, acc, mask=k_mask)


@triton.jit
def _reduce_sq_diff_2d_contig_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    red_size,
    red_stride,
    base_stride_0,
    out_dim1,
    out_stride_0,
    correction,
    BLOCK_SIZE: tl.constexpr,
):
    """Variance for 2D output with contiguous memory access.
    Each block handles one value of output dim 0 with stride-1 loads."""
    pid = tl.program_id(0)
    base = pid * base_stride_0

    for k_start in range(0, out_dim1, BLOCK_SIZE):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offs < out_dim1
        mean_offs = pid * out_stride_0 + k_start + tl.arange(0, BLOCK_SIZE)
        mean_vals = tl.load(mean_ptr + mean_offs, mask=k_mask, other=0.0)

        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for j in tl.range(0, red_size, 1):
            vals = tl.load(
                x_ptr + base + j * red_stride + k_offs, mask=k_mask, other=0.0
            )
            diff = vals.to(tl.float32) - mean_vals
            acc += diff * diff

        n = tl.maximum(red_size - correction, 1)
        var = acc / n.to(tl.float32)
        out_offs = pid * out_stride_0 + k_start + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + out_offs, var, mask=k_mask)


@triton.jit
def _reduce_partial_mean_var_kernel(
    partial_sum_ptr,
    partial_sq_ptr,
    out_ptr,
    num_chunks,
    red_size,
    correction,
    out_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce partial sum and sum_sq and compute mean and variance."""
    pid = tl.program_id(0)
    if pid >= out_numel:
        return

    acc_sum = tl.zeros([1], dtype=tl.float32)
    acc_sq = tl.zeros([1], dtype=tl.float32)
    for chunk_idx in tl.range(0, num_chunks, BLOCK_SIZE):
        chunk_offs = chunk_idx + tl.arange(0, BLOCK_SIZE)
        chunk_mask = chunk_offs < num_chunks
        vals_s = tl.load(
            partial_sum_ptr + pid * num_chunks + chunk_offs,
            mask=chunk_mask,
            other=0.0,
        )
        vals_q = tl.load(
            partial_sq_ptr + pid * num_chunks + chunk_offs,
            mask=chunk_mask,
            other=0.0,
        )
        acc_sum += tl.sum(vals_s)
        acc_sq += tl.sum(vals_q)

    total_sum = tl.sum(acc_sum)
    total_sq = tl.sum(acc_sq)
    mean = total_sum / red_size
    var_pop = total_sq / red_size - mean * mean
    n_adj = red_size - correction
    if n_adj < 1:
        n_adj = 1
    var = var_pop * red_size / n_adj
    tl.store(out_ptr + pid, var)


@triton.jit
def _partial_sum_sq_chunk_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sq_ptr,
    red_size,
    red_stride,
    base_stride_0,
    num_chunks,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute partial sum and sum of squares for one chunk."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    base = pid0 * base_stride_0
    chunk_start = pid1 * BLOCK_SIZE

    offs = chunk_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < red_size
    vals = tl.load(x_ptr + base + offs * red_stride, mask=mask, other=0.0)
    vals_f32 = vals.to(tl.float32)
    partial_sum = tl.sum(vals_f32)
    partial_sq = tl.sum(vals_f32 * vals_f32)
    tl.store(partial_sum_ptr + pid0 * num_chunks + pid1, partial_sum)
    tl.store(partial_sq_ptr + pid0 * num_chunks + pid1, partial_sq)


@triton.jit
def _reduce_sum_sq_2d_contig_kernel(
    x_ptr,
    sum_ptr,
    sq_ptr,
    red_size,
    red_stride,
    base_stride_0,
    out_dim1,
    out_stride_0,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused sum and sum-of-squares for 2D output with contiguous access."""
    pid = tl.program_id(0)
    base = pid * base_stride_0

    for k_start in range(0, out_dim1, BLOCK_SIZE):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offs < out_dim1
        acc_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for j in tl.range(0, red_size, 1):
            vals = tl.load(
                x_ptr + base + j * red_stride + k_offs, mask=k_mask, other=0.0
            )
            vals_f32 = vals.to(tl.float32)
            acc_sum += vals_f32
            acc_sq += vals_f32 * vals_f32

        out_offs = pid * out_stride_0 + k_start + tl.arange(0, BLOCK_SIZE)
        tl.store(sum_ptr + out_offs, acc_sum, mask=k_mask)
        tl.store(sq_ptr + out_offs, acc_sq, mask=k_mask)


@triton.jit
def _mean_var_from_sum_sq_2d_kernel(
    sum_ptr,
    sq_ptr,
    out_ptr,
    red_size,
    correction,
    out_dim1,
    out_stride_0,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute mean and variance from sum and sum_sq for 2D output."""
    pid = tl.program_id(0)

    for k_start in range(0, out_dim1, BLOCK_SIZE):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offs < out_dim1
        out_offs = pid * out_stride_0 + k_start + tl.arange(0, BLOCK_SIZE)

        s = tl.load(sum_ptr + out_offs, mask=k_mask, other=0.0)
        sq = tl.load(sq_ptr + out_offs, mask=k_mask, other=0.0)

        mean = s / red_size
        var_pop = sq / red_size - mean * mean
        n_adj = red_size - correction
        if n_adj < 1:
            n_adj = 1
        var = var_pop * red_size / n_adj
        tl.store(out_ptr + out_offs, var, mask=k_mask)


@triton.jit
def _partial_sum_chunk_kernel(
    x_ptr,
    partial_out_ptr,
    red_size,
    red_stride,
    base_stride_0,
    num_chunks,
    BLOCK_SIZE: tl.constexpr,
):
    """Partial sum: each block sums one chunk of the reduction dimension."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    base = pid0 * base_stride_0
    chunk_start = pid1 * BLOCK_SIZE

    offs = chunk_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < red_size
    vals = tl.load(x_ptr + base + offs * red_stride, mask=mask, other=0.0)
    partial_sum = tl.sum(vals.to(tl.float32))
    tl.store(partial_out_ptr + pid0 * num_chunks + pid1, partial_sum)


@triton.jit
def _reduce_partial_sum_kernel(
    partial_in_ptr,
    out_ptr,
    num_chunks,
    divisor,
    out_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce partial sums and divide by divisor to get mean."""
    pid = tl.program_id(0)
    if pid >= out_numel:
        return

    acc = tl.zeros([1], dtype=tl.float32)
    for chunk_idx in tl.range(0, num_chunks, BLOCK_SIZE):
        chunk_offs = chunk_idx + tl.arange(0, BLOCK_SIZE)
        chunk_mask = chunk_offs < num_chunks
        vals = tl.load(
            partial_in_ptr + pid * num_chunks + chunk_offs,
            mask=chunk_mask,
            other=0.0,
        )
        acc += tl.sum(vals)

    acc_sum = tl.sum(acc)
    tl.store(out_ptr + pid, acc_sum / divisor)


@triton.jit
def _partial_sq_diff_chunk_kernel(
    x_ptr,
    mean_ptr,
    partial_out_ptr,
    red_size,
    red_stride,
    base_stride_0,
    num_chunks,
    BLOCK_SIZE: tl.constexpr,
):
    """Partial squared diff: each block sums (x-mean)^2 for one chunk."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    base = pid0 * base_stride_0
    mean_val = tl.load(mean_ptr + pid0)
    chunk_start = pid1 * BLOCK_SIZE

    offs = chunk_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < red_size
    vals = tl.load(x_ptr + base + offs * red_stride, mask=mask, other=mean_val)
    diff = vals.to(tl.float32) - mean_val
    partial_sq = tl.sum(diff * diff)
    tl.store(partial_out_ptr + pid0 * num_chunks + pid1, partial_sq)


@triton.jit
def _reduce_partial_var_kernel(
    partial_in_ptr,
    out_ptr,
    num_chunks,
    n,
    out_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce partial squared diffs and divide by n to get variance."""
    pid = tl.program_id(0)
    if pid >= out_numel:
        return

    acc = tl.zeros([1], dtype=tl.float32)
    for chunk_idx in tl.range(0, num_chunks, BLOCK_SIZE):
        chunk_offs = chunk_idx + tl.arange(0, BLOCK_SIZE)
        chunk_mask = chunk_offs < num_chunks
        vals = tl.load(
            partial_in_ptr + pid * num_chunks + chunk_offs,
            mask=chunk_mask,
            other=0.0,
        )
        acc += tl.sum(vals)

    acc_sum = tl.sum(acc)
    tl.store(out_ptr + pid, acc_sum / n)


def var(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ASCEND VAR")
    if correction is None:
        correction = 1

    # --- Parse dim ---
    ndim = x.ndim
    if dim is None:
        dims_tuple = tuple(range(ndim))
        n_red_dims = ndim
    elif dim * 0 == 0:
        dims_tuple = (dim % ndim,)
        n_red_dims = 1
    else:
        n_red_dims = len(dim)
        dims_tuple = ()
        for i in range(n_red_dims):
            dims_tuple = dims_tuple + (dim[i] % ndim,)

    shape = x.shape

    # --- Compute C-order strides (up to 4D) ---
    if ndim == 1:
        strides = (1,)
    elif ndim == 2:
        strides = (shape[1], 1)
    elif ndim == 3:
        strides = (shape[1] * shape[2], shape[2], 1)
    else:
        strides = (
            shape[1] * shape[2] * shape[3],
            shape[2] * shape[3],
            shape[3],
            1,
        )

    # --- Compute output shape ---
    if keepdim:
        out_shape = ()
        for i in range(ndim):
            is_red = False
            for j in range(n_red_dims):
                if dims_tuple[j] == i:
                    is_red = True
            if is_red:
                out_shape = out_shape + (1,)
            else:
                out_shape = out_shape + (shape[i],)
    else:
        out_shape = ()
        for i in range(ndim):
            is_red = False
            for j in range(n_red_dims):
                if dims_tuple[j] == i:
                    is_red = True
            if not is_red:
                out_shape = out_shape + (shape[i],)

    # --- Compute output numel ---
    out_numel = 1
    for i in range(len(out_shape)):
        out_numel *= out_shape[i]
    if out_numel == 0:
        out_numel = 1

    # --- Compute output strides (C-order, up to 3D) ---
    out_ndim = len(out_shape)
    if out_ndim == 0:
        out_strides = (0,)
    elif out_ndim == 1:
        out_strides = (1,)
    elif out_ndim == 2:
        out_strides = (out_shape[1], 1)
    else:
        out_strides = (out_shape[1] * out_shape[2], out_shape[2], 1)

    # --- Reduction size ---
    red_size = 1
    for i in range(n_red_dims):
        red_size *= shape[dims_tuple[i]]

    out_dtype = x.dtype

    # Allocate mean intermediate (float32 for precision) and output
    mean_interm = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    out = torch.empty(out_shape, dtype=out_dtype, device=x.device)

    BLOCK_SIZE = 256
    PARTIAL_BLOCK_SIZE = 1024

    if n_red_dims == 1:
        red_dim = dims_tuple[0]
        red_stride = strides[red_dim]

        if len(out_shape) <= 1:
            # 1D output grid
            if out_numel == 1:
                base_stride_0 = 0
            elif red_dim != 0:
                base_stride_0 = strides[0]
            else:
                base_stride_0 = 1

            if out_numel > 1 and red_dim != ndim - 1:
                kept_count = 0
                kept_first = -1
                for i in range(ndim):
                    if i != red_dim:
                        kept_count = kept_count + 1
                        if kept_count == 1:
                            kept_first = i
                if kept_count == 1 and kept_first >= 0:
                    base_stride_0 = strides[kept_first]

            # Determine if partial (multi-chunk) reduction is beneficial
            num_chunks = triton.cdiv(red_size, PARTIAL_BLOCK_SIZE)

            if num_chunks > 32 and out_numel * num_chunks < 65536:
                # Partial reduction with fused sum+sum_sq in one pass
                partial_shape = (out_numel, num_chunks)
                partial_sum_tensor = torch.empty(
                    partial_shape, dtype=torch.float32, device=x.device
                )
                partial_sq_tensor = torch.empty(
                    partial_shape, dtype=torch.float32, device=x.device
                )

                partial_grid = (out_numel, num_chunks)
                _partial_sum_sq_chunk_kernel[partial_grid](
                    x,
                    partial_sum_tensor,
                    partial_sq_tensor,
                    red_size,
                    red_stride,
                    base_stride_0,
                    num_chunks,
                    BLOCK_SIZE=PARTIAL_BLOCK_SIZE,
                )

                reduce_blk = 128
                reduce_grid = (out_numel,)
                _reduce_partial_mean_var_kernel[reduce_grid](
                    partial_sum_tensor,
                    partial_sq_tensor,
                    out,
                    num_chunks,
                    red_size,
                    correction,
                    out_numel,
                    BLOCK_SIZE=reduce_blk,
                )

                return out
            else:
                # Single chunk: use fused sum+sum_sq approach
                grid = (max(out_numel, 1),)
                out_stride_0_val = 1 if out_numel > 1 else 0

                sum_interm_single = torch.empty(
                    out_shape, dtype=torch.float32, device=x.device
                )
                sq_interm_single = torch.empty(
                    out_shape, dtype=torch.float32, device=x.device
                )

                _reduce_sum_sq_1d_kernel[grid](
                    x,
                    sum_interm_single,
                    sq_interm_single,
                    red_size,
                    red_stride,
                    base_stride_0,
                    out_stride_0_val,
                    BLOCK_SIZE=PARTIAL_BLOCK_SIZE,
                )

                mean_var_grid = (triton.cdiv(out_numel, PARTIAL_BLOCK_SIZE),)
                _mean_var_from_sum_sq_1d_kernel[mean_var_grid](
                    sum_interm_single,
                    sq_interm_single,
                    out,
                    red_size,
                    correction,
                    out_numel,
                    BLOCK_SIZE=PARTIAL_BLOCK_SIZE,
                )
        else:
            # 2D output grid - use contiguous memory access, fused sum+sum_sq
            grid = (out_shape[0],)

            if keepdim:
                base_stride_0 = 0 if dims_tuple[0] == 0 else strides[0]
            else:
                kept_vals = ()
                for i in range(ndim):
                    if i != red_dim:
                        kept_vals = kept_vals + (strides[i],)
                base_stride_0 = kept_vals[0]

            out_stride_0 = out_strides[0] if len(out_strides) > 0 else 0

            # Fused: compute sum and sum_sq in one pass
            sum_interm = torch.empty(out_shape, dtype=torch.float32, device=x.device)
            sq_interm = torch.empty(out_shape, dtype=torch.float32, device=x.device)

            _reduce_sum_sq_2d_contig_kernel[grid](
                x,
                sum_interm,
                sq_interm,
                red_size,
                red_stride,
                base_stride_0,
                out_shape[1],
                out_stride_0,
                BLOCK_SIZE=PARTIAL_BLOCK_SIZE,
            )

            # Compute mean and variance from sum and sum_sq
            _mean_var_from_sum_sq_2d_kernel[grid](
                sum_interm,
                sq_interm,
                out,
                red_size,
                correction,
                out_shape[1],
                out_stride_0,
                BLOCK_SIZE=PARTIAL_BLOCK_SIZE,
            )

    else:
        if len(out_shape) <= 1:
            grid = (max(out_numel, 1),)

            red_dim0 = dims_tuple[0]
            red_dim1 = dims_tuple[1]
            red_stride_0 = strides[red_dim0]
            red_stride_1 = strides[red_dim1]
            red_dim1_size = shape[red_dim1]

            if out_numel == 1:
                base_stride_0 = 0
            else:
                kept_vals = ()
                for i in range(ndim):
                    is_red = False
                    for j in range(n_red_dims):
                        if dims_tuple[j] == i:
                            is_red = True
                    if not is_red:
                        kept_vals = kept_vals + (strides[i],)
                if len(kept_vals) == 1:
                    base_stride_0 = kept_vals[0]
                else:
                    base_stride_0 = 1

            out_stride_0_val = 1 if out_numel > 1 else 0

            _reduce_sum_multi_dim_1d_kernel[grid](
                x,
                mean_interm,
                red_size,
                red_dim1_size,
                red_stride_0,
                red_stride_1,
                base_stride_0,
                out_stride_0_val,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            div_grid = (triton.cdiv(out_numel, BLOCK_SIZE),)
            _div_scalar_kernel[div_grid](
                mean_interm, out_numel, red_size, BLOCK_SIZE=BLOCK_SIZE
            )

            _reduce_sq_diff_multi_dim_1d_kernel[grid](
                x,
                mean_interm,
                out,
                red_size,
                red_dim1_size,
                red_stride_0,
                red_stride_1,
                base_stride_0,
                out_stride_0_val,
                correction,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            grid = (out_numel,)

            red_dim0 = dims_tuple[0]
            red_dim1 = dims_tuple[1]
            red_stride_0 = strides[red_dim0]
            red_stride_1 = strides[red_dim1]
            red_dim1_size = shape[red_dim1]

            base_stride_0 = 1

            _reduce_sum_multi_dim_1d_kernel[grid](
                x,
                mean_interm,
                red_size,
                red_dim1_size,
                red_stride_0,
                red_stride_1,
                base_stride_0,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            div_grid = (triton.cdiv(out_numel, BLOCK_SIZE),)
            _div_scalar_kernel[div_grid](
                mean_interm, out_numel, red_size, BLOCK_SIZE=BLOCK_SIZE
            )

            _reduce_sq_diff_multi_dim_1d_kernel[grid](
                x,
                mean_interm,
                out,
                red_size,
                red_dim1_size,
                red_stride_0,
                red_stride_1,
                base_stride_0,
                1,
                correction,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out


def var_dim(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ASCEND VAR_DIM")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ASCEND VAR_CORRECTION")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)
