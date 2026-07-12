import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_gammaincc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_gammaincc(shape, dtype):
    # Use positive values only since gammaincc requires positive inputs
    # float16 and bfloat16 are not supported by torch.special.gammaincc on CUDA
    inp1 = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5 + 0.1
    inp2 = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5 + 0.1
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.special.gammaincc(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.special.gammaincc(inp1, inp2)

    # Use a more lenient tolerance (atol=1e-2) for this complex mathematical function
    flag_gems.testing.assert_close(res_out, ref_out, dtype, atol=1e-2)


@pytest.mark.special_gammaincc
def test_gammaincc_boundary_values():
    dtype = torch.float32
    inp1 = torch.tensor(
        [0.0, -1.0, 1.0, 1.0, 0.5, 2.5, 10.0, math.inf, math.nan],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp2 = torch.tensor(
        [1.0, 1.0, -1.0, 0.0, 0.25, 5.0, math.inf, 1.0, 1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.special.gammaincc(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.special.gammaincc(inp1, inp2)

    flag_gems.testing.assert_close(res_out, ref_out, dtype, equal_nan=True, atol=1e-2)
