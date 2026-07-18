import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_expm1
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_expm1(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.expm1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.expm1(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_expm1
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_expm1_near_zero(dtype):
    inp = torch.tensor(
        [-1.0e-4, -1.0e-7, 0.0, 1.0e-7, 1.0e-4],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.expm1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.expm1(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
