import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.lcm
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_lcm(shape, dtype):
    inp1 = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.lcm(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.lcm(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)
