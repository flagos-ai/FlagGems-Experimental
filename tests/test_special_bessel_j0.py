import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_bessel_j0
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int32])
def test_special_bessel_j0(shape, dtype):
    if dtype.is_floating_point:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-16, 16, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.special.bessel_j0(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.bessel_j0(inp)
    utils.gems_assert_close(res_out, ref_out, ref_out.dtype)


@pytest.mark.special_bessel_j0
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_bessel_j0_special_values(dtype):
    values = torch.tensor(
        [0.0, 1.0, 3.8317059702075125, 8.0, -8.0, float("nan"), float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values)
    ref_out = torch.special.bessel_j0(ref_values)
    with flag_gems.use_gems():
        res_out = torch.special.bessel_j0(values)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_bessel_j0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_special_bessel_j0_unsupported_cuda_dtypes(dtype):
    inp = torch.randn((8,), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="bessel_j0_cuda"):
        with flag_gems.use_gems():
            torch.special.bessel_j0(inp)
