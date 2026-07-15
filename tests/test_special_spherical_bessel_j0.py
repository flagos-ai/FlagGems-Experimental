import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_spherical_bessel_j0
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_spherical_bessel_j0(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.spherical_bessel_j0(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.spherical_bessel_j0(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_spherical_bessel_j0_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# torch.special.spherical_bessel_j0 out= variant only dispatches for float32
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_spherical_bessel_j0_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = torch.special.spherical_bessel_j0(ref_inp)
    ref_inp.copy_(ref_out)
    with flag_gems.use_gems():
        res_out = torch.special.spherical_bessel_j0(inp, out=inp)
    utils.gems_assert_close(inp, ref_inp, dtype)
    utils.gems_assert_close(res_out, ref_inp, dtype)
