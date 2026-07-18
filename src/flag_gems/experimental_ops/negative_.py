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

import torch  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def negative_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = -x
    tl.store(x_ptr + offsets, x, mask=mask)


_negative__kernel = negative_


def negative_(*args, **kwargs):
    x = args[0] if len(args) > 0 else kwargs.get("input", kwargs.get("self", None))
    if x is None:
        raise ValueError("negative_ expects a tensor as the first argument")
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    n_elements = x.numel()
    if n_elements == 0:
        return x
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _negative__kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
