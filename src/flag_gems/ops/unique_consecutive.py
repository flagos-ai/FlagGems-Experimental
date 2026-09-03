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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def unique_consecutive_mask_kernel(
    inp,
    mask_out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to identify positions where consecutive elements differ.
    mask_out[i] = 1 if inp[i] != inp[i-1] (or i==0), else 0
    """
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load current elements
    curr_ptrs = inp + offset
    curr_vals = tl.load(curr_ptrs, mask=mask, other=0)

    # Load previous elements (shifted by 1)
    prev_offset = offset - 1
    prev_mask = (offset > 0) & (offset < n_elements)
    prev_ptrs = inp + prev_offset
    prev_vals = tl.load(prev_ptrs, mask=prev_mask, other=0)

    # First element is always unique, others are unique if different from previous
    is_first = offset == 0
    is_different = curr_vals != prev_vals
    is_unique = is_first | (is_different & mask)

    # Store mask as int32
    out_ptrs = mask_out + offset
    tl.store(out_ptrs, is_unique.to(tl.int32), mask=mask)


@libentry()
@triton.jit
def unique_consecutive_mask_dim_kernel(
    inp,
    mask_out,
    dim_size,
    stride,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to identify positions along a dimension where consecutive elements differ.
    For each row, mask_out[i] = 1 if row[i] != row[i-1] (or i==0), else 0
    """
    pid = ext.program_id(0)
    row_idx = pid

    if row_idx >= n_rows:
        return

    base_offset = row_idx * stride * dim_size

    for i in range(0, dim_size, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < dim_size

        # Load current elements along the dimension
        curr_ptrs = inp + base_offset + offset * stride
        curr_vals = tl.load(curr_ptrs, mask=mask, other=0)

        # Load previous elements
        prev_offset = offset - 1
        prev_mask = (offset > 0) & (offset < dim_size)
        prev_ptrs = inp + base_offset + prev_offset * stride
        prev_vals = tl.load(prev_ptrs, mask=prev_mask, other=0)

        # First element is always unique, others are unique if different from previous
        is_first = offset == 0
        is_different = curr_vals != prev_vals
        is_unique = is_first | (is_different & mask)

        # Store mask
        out_ptrs = mask_out + row_idx * dim_size + offset
        tl.store(out_ptrs, is_unique.to(tl.int32), mask=mask)


def unique_consecutive(inp, return_inverse=False, return_counts=False, dim=None):
    """
    Eliminates all but the first element from every consecutive group of equivalent elements.

    Args:
        inp: Input tensor
        return_inverse: Whether to return inverse indices
        return_counts: Whether to return counts of each unique element
        dim: Dimension along which to apply unique. If None, flattens the input.

    Returns:
        Tuple of (output, inverse_indices, counts)
    """
    logger.debug("GEMS UNIQUE_CONSECUTIVE")

    if dim is None:
        # Flatten case
        inp_flat = inp.flatten()
        n_elements = inp_flat.numel()

        if n_elements == 0:
            # Empty tensor case
            empty_out = torch.empty(0, dtype=inp.dtype, device=inp.device)
            empty_inverse = torch.empty(0, dtype=torch.long, device=inp.device)
            empty_counts = torch.empty(0, dtype=torch.long, device=inp.device)
            return empty_out, empty_inverse, empty_counts

        if n_elements == 1:
            # Single element case
            inverse = (
                torch.zeros(1, dtype=torch.long, device=inp.device)
                if return_inverse
                else torch.empty(0, dtype=torch.long, device=inp.device)
            )
            counts = (
                torch.ones(1, dtype=torch.long, device=inp.device)
                if return_counts
                else torch.empty(0, dtype=torch.long, device=inp.device)
            )
            return inp_flat, inverse, counts

        # Create mask indicating where consecutive elements differ
        mask = torch.empty(n_elements, dtype=torch.int32, device=inp.device)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        with torch_device_fn.device(inp.device):
            unique_consecutive_mask_kernel[grid](inp_flat, mask, n_elements, BLOCK_SIZE)

        # Convert mask to bool
        mask_bool = mask.to(torch.bool)

        # Get unique values - avoid dispatch to potentially buggy nonzero
        # Use boolean indexing which is more efficient
        output = inp_flat[mask_bool]

        # Compute inverse indices if requested
        if return_inverse:
            # Use cumsum on mask to get group indices
            inverse = torch.cumsum(mask_bool.to(torch.long), dim=0) - 1
        else:
            inverse = torch.empty(0, dtype=torch.long, device=inp.device)

        # Compute counts if requested
        if return_counts:
            # Count elements in each group
            n_unique = output.numel()
            if n_unique > 0:
                # Append sentinel to handle last group
                group_ids = torch.cumsum(mask_bool.to(torch.long), dim=0) - 1
                counts = torch.zeros(n_unique, dtype=torch.long, device=inp.device)
                counts.scatter_add_(0, group_ids, torch.ones_like(group_ids))
            else:
                counts = torch.empty(0, dtype=torch.long, device=inp.device)
        else:
            counts = torch.empty(0, dtype=torch.long, device=inp.device)

        return output, inverse, counts

    else:
        # For the dim case, fall back to PyTorch implementation for now
        # This is a complex operation that requires comparing entire slices
        # A full Triton implementation would be very involved
        result = torch.unique_consecutive(
            inp, return_inverse=return_inverse, return_counts=return_counts, dim=dim
        )
        return result


def unique_consecutive_out(
    inp, return_inverse=False, return_counts=False, dim=None, *, out0, out1, out2
):
    """
    Output variant of unique_consecutive.
    """
    logger.debug("GEMS UNIQUE_CONSECUTIVE_OUT")

    result_out, result_inverse, result_counts = unique_consecutive(
        inp, return_inverse, return_counts, dim
    )

    # Copy to output tensors
    out0.resize_(result_out.shape).copy_(result_out)
    out1.resize_(result_inverse.shape).copy_(result_inverse)
    out2.resize_(result_counts.shape).copy_(result_counts)

    return out0, out1, out2
