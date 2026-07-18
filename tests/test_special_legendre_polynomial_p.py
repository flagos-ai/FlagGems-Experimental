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


@pytest.mark.special_legendre_polynomial_p
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_legendre_polynomial_p(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # n is the degree of the polynomial, typically small (0-10)
    # Use small integer values for n
    n_elements = inp.numel()
    n = torch.randint(0, 10, (n_elements,), dtype=torch.long, device=flag_gems.device)
    n = n.reshape(inp.shape)
    ref_n = utils.to_reference(n)

    # PyTorch's legendre_polynomial_p doesn't support float16/bfloat16,
    # so compute reference in float32 and convert back
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.special.legendre_polynomial_p(ref_inp.float(), ref_n).to(dtype)
    else:
        ref_out = torch.special.legendre_polynomial_p(ref_inp, ref_n)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_legendre_polynomial_p(inp, n)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_legendre_polynomial_p
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_legendre_polynomial_p_scalar_n(shape, dtype):
    # Test with scalar n (same degree for all elements)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Use a single scalar n value
    n = 3  # polynomial degree
    ref_n = n

    # PyTorch's legendre_polynomial_p doesn't support float16/bfloat16,
    # so compute reference in float32 and convert back
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.special.legendre_polynomial_p(ref_inp.float(), ref_n).to(dtype)
    else:
        ref_out = torch.special.legendre_polynomial_p(ref_inp, ref_n)
    # Convert n to tensor for aten op
    n_tensor = torch.tensor(n, dtype=torch.long, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_legendre_polynomial_p(inp, n_tensor)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_legendre_polynomial_p
def test_special_legendre_polynomial_p_broadcast_and_large_degree():
    inp = torch.tensor(
        [[-0.75, -0.25, 0.2], [0.4, 0.8, 0.95]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    n = torch.tensor([0, 3, 101], dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_n = utils.to_reference(n)

    ref_out = torch.special.legendre_polynomial_p(ref_inp, ref_n)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_legendre_polynomial_p(inp, n)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.special_legendre_polynomial_p
def test_special_legendre_polynomial_p_out_broadcast_resize():
    inp = torch.tensor(
        [[-0.5, 0.0, 0.5], [0.25, 0.75, 1.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    n = torch.tensor([[2], [101]], dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_n = utils.to_reference(n)
    out = torch.empty((1,), dtype=torch.float32, device=flag_gems.device)

    ref_out = torch.special.legendre_polynomial_p(ref_inp, ref_n)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_legendre_polynomial_p.out(inp, n, out=out)

    assert res_out is out
    assert out.shape == ref_out.shape
    utils.gems_assert_close(out, ref_out, torch.float32)
