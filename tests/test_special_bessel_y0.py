import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_bessel_y0
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_bessel_y0(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device).abs() + 0.1
    boundary = torch.tensor(
        [0.0, -1.0, -0.1, 1.0e-4, 1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    x.reshape(-1)[: boundary.numel()] = boundary
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_bessel_y0(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.special_bessel_y0(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_bessel_y0(x)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_bessel_y0
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_bessel_y0_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device).abs() + 0.1
    boundary = torch.tensor(
        [0.0, -1.0, -0.1, 1.0e-4, 1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    x.reshape(-1)[: boundary.numel()] = boundary
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        out_ref = torch.empty_like(ref_x, dtype=torch.float32)
        ref_out = torch.ops.aten.special_bessel_y0.out(ref_x.float(), out=out_ref)
        out_ref = out_ref.to(dtype)
        ref_out = out_ref
    else:
        out_ref = torch.empty_like(ref_x)
        ref_out = torch.ops.aten.special_bessel_y0.out(ref_x, out=out_ref)
    out_act = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_bessel_y0.out(x, out=out_act)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(out_act, out_ref, dtype, equal_nan=True)
