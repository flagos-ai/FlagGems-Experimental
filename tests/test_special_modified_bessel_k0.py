import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_modified_bessel_k0
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_modified_bessel_k0(shape, dtype):
    # K0 is only defined for x > 0, so generate positive inputs only
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.modified_bessel_k0(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.modified_bessel_k0(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_modified_bessel_k0_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_modified_bessel_k0_out(shape, dtype):
    # K0 is only defined for x > 0, so generate positive inputs only
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.modified_bessel_k0(ref_inp)
    with flag_gems.use_gems():
        torch.special.modified_bessel_k0(inp, out=out)
    utils.gems_assert_close(out, ref_out, dtype)
