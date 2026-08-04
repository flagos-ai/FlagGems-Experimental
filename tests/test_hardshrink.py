import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.hardshrink
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("lambd", [0.0, 0.5, 1.0])
def test_hardshrink(shape, dtype, lambd):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten.hardshrink(ref_inp, lambd)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardshrink(res_inp, lambd)

    utils.gems_assert_close(res_out, ref_out, dtype)
