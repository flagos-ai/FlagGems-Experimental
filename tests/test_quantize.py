import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.quantize
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    "zero_point, quant_min, quant_max",
    [(10, 0, 255), (0, -128, 127)],
)
def test_quantize(shape, dtype, zero_point, quant_min, quant_max):
    scale = 0.1

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clamp(
        torch.round(ref_inp / scale + zero_point), quant_min, quant_max
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = flag_gems.quantize(
            inp, scale, zero_point, quant_min=quant_min, quant_max=quant_max
        )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1)
