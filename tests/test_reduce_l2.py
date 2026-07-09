import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reduce_l2
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + [(0,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reduce_l2(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute L2 norm: sqrt(sum(x^2))
    ref_out = torch.linalg.vector_norm(ref_inp, ord=2)
    with flag_gems.use_gems():
        res_out = flag_gems.reduce_l2(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
