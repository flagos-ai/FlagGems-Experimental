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

# Hygon specialization of ``aten::dequantize``.
#
# The only change against the generic kernel is how the integer payload is
# obtained: ``qint8`` storage is already int8, so ``Tensor.view(torch.int8)``
# hands the kernel the existing device buffer instead of materialising a
# second one with ``int_repr()``.  ``quint8``/``qint32`` keep ``int_repr()``
# because reinterpreting their storage as int8 would change the stored values
# (the payload is read as a signed int8 inside the kernel).
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024


@triton.jit
def dequantize_kernel(
    x, scale, zero_point, output, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_val = tl.load(x + offsets, mask=mask, other=0).to(tl.float32)
    result = (x_val - zero_point) * scale
    tl.store(output + offsets, result, mask=mask)


def dequantize(a):
    """Dequantize a per-tensor quantized tensor into float32.

    Args:
        a: A quantized tensor (e.g., torch.qint8, torch.quint8)

    Returns:
        A float32 tensor with the dequantized values
    """
    logger.debug("GEMS DEQUANTIZE")

    # Zero-copy reinterpret of the qint8 payload: same storage as int_repr(),
    # without materializing a second device buffer.
    if a.dtype is torch.qint8:
        payload = a.view(torch.int8)
    else:
        payload = a.int_repr().to(a.device)
    scale = float(a.q_scale())
    zero_point = int(a.q_zero_point())

    output = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)

    with torch_device_fn.device(a.device):
        dequantize_kernel[grid](
            payload, scale, zero_point, output, n_elements, _BLOCK_SIZE
        )
    return output
