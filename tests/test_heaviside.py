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


@pytest.mark.heaviside
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_heaviside(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < 0.1
    inp[mask] = 0.0

    ref_inp = utils.to_reference(inp, True)
    ref_values = utils.to_reference(values, True)
    ref_out = torch.heaviside(ref_inp, ref_values)

    with flag_gems.use_gems():
        res_out = torch.heaviside(inp, values)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.heaviside_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_heaviside_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < 0.1
    inp[mask] = 0.0

    ref_inp = utils.to_reference(inp, True)
    ref_values = utils.to_reference(values, True)
    ref_out_buf = torch.empty_like(ref_inp)
    ref_out = torch.ops.aten.heaviside.out(ref_inp, ref_values, out=ref_out_buf)

    res_out_buf = torch.empty_like(inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.heaviside.out(inp, values, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
