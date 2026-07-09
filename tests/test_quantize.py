import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.quantize
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_quantize(shape, dtype):
    scale = 0.1
    zero_point = 10

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clamp(torch.round(ref_inp / scale + zero_point), 0, 255).to(dtype)

    with flag_gems.use_gems():
        res_out = flag_gems.quantize(inp, scale, zero_point)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1)
