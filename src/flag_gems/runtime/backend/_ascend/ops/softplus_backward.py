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
def _softplus_backward_kernel(
    grad_output_ptr,
    self_input_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles NUM_BLOCKS_PER_PROGRAM blocks
    for blk in range(NUM_BLOCKS_PER_PROGRAM):
        block_id = pid * NUM_BLOCKS_PER_PROGRAM + blk
        block_start = block_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        grad = tl.load(grad_output_ptr + offsets, mask=mask)
        x = tl.load(self_input_ptr + offsets, mask=mask)

        # Use float32 intermediate precision for numerical accuracy
        grad_f32 = grad.to(tl.float32)
        x_f32 = x.to(tl.float32)

        beta_x = beta * x_f32

        # Compute sigmoid(beta_x) and apply threshold
        # Using tl.sigmoid for potentially optimized implementation
        sigmoid_val = tl.sigmoid(beta_x)

        use_threshold = beta_x >= threshold
        result_f32 = tl.where(use_threshold, grad_f32, grad_f32 * sigmoid_val)

        # Cast back to original dtype
        result = result_f32.to(grad.dtype)

        tl.store(out_ptr + offsets, result, mask=mask)


def softplus_backward(grad_output, self_input, beta, threshold):
    logger.debug("GEMS_ASCEND SOFTPLUS_BACKWARD")
    n_elements = grad_output.numel()

    out = torch.empty_like(grad_output)

    MAX_GRID = 65535

    if n_elements <= 4096:
        BLOCK_SIZE = max(1024, n_elements)
        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    else:
        BLOCK_SIZE = 4096
        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    grid_size = min(num_blocks, MAX_GRID)
    num_blocks_per_program = triton.cdiv(num_blocks, grid_size)
    grid = (grid_size,)

    _softplus_backward_kernel[grid](
        grad_output,
        self_input,
        out,
        n_elements,
        beta=beta,
        threshold=threshold,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_BLOCKS_PER_PROGRAM=num_blocks_per_program,
    )

    return out
