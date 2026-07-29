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

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def uniform_kernel(
    x_ptr,
    n_elements,
    from_val,
    to_val,
    seed,
    BLOCK_SIZE: tl.constexpr,
    N_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)

    for i in range(N_BLOCKS):
        block_id = pid + i * grid_size
        block_start = block_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Fast PRNG: multiplicative hash with golden-ratio conjugate
        # x * phi mod 1 gives well-distributed pseudo-random values in [0, 1)
        x = offsets * 0.6180339887498949 + seed
        rand_vals = x - tl.floor(x)

        scale = to_val - from_val
        vals = from_val + rand_vals * scale

        tl.store(x_ptr + offsets, vals, mask=mask)


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS_ASCEND UNIFORM_")
    n_elements = self.numel()
    BLOCK_SIZE = 10240
    MAX_GRID = 65535
    total_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    grid_size = min(MAX_GRID, total_blocks)
    n_blocks = triton.cdiv(total_blocks, grid_size)
    grid = (grid_size,)
    seed = 42
    uniform_kernel[grid](
        self,
        n_elements,
        from_,
        to,
        seed,
        BLOCK_SIZE=BLOCK_SIZE,
        N_BLOCKS=n_blocks,
    )
    return self
