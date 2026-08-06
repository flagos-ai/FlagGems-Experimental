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
from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

_cyl_bessel_i1 = tl_extra_shim.cyl_bessel_i1


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_i1_func(x):
    # Compute in float32 for numerical stability, using the device's native
    # modified Bessel function I1 intrinsic. This is both faster and more
    # accurate than the Cephes polynomial approximation used in the generic
    # implementation, and pointwise_dynamic handles non-contiguous inputs and
    # autotunes launch parameters.
    x_f32 = x.to(tl.float32)
    return _cyl_bessel_i1(x_f32)


def special_i1(self: torch.Tensor):
    logger.debug("GEMS_HYGON SPECIAL_I1")
    return special_i1_func(self)


def special_i1_out(self: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_HYGON SPECIAL_I1_OUT")
    if out.dtype != self.dtype:
        raise TypeError("out dtype must match input dtype")
    if out.device != self.device:
        raise TypeError("out device must match input device")
    return special_i1_func(self, out0=out)
