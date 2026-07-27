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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.heaviside_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_heaviside_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < 0.1
    inp[mask] = 0.0

    ref_inp = utils.to_reference(inp.clone(), True)
    ref_values = utils.to_reference(values, True)
    ref_out = ref_inp.heaviside_(ref_values)

    with flag_gems.use_gems():
        res_out = inp.heaviside_(values)

    utils.gems_assert_close(res_out, ref_out, dtype)
