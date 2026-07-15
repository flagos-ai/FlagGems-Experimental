import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

MULTIGAMMALN_P_VALUES = [1, 2, 3, 4, 5]


def _get_multigammaln_input(shape, dtype, device, p):
    """Generate valid input for multigammaln: all elements must be > (p-1)/2"""
    min_val = (p - 1) / 2 + 0.1  # ensure strict inequality
    return torch.rand(shape, dtype=dtype, device=flag_gems.device) + min_val


@pytest.mark.special_multigammaln
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", MULTIGAMMALN_P_VALUES)
def test_special_multigammaln(shape, dtype, p):
    x = _get_multigammaln_input(shape, dtype, flag_gems.device, p)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_multigammaln(ref_x.float(), p).to(dtype)
    else:
        ref_out = torch.ops.aten.special_multigammaln(ref_x, p)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_multigammaln(x, p)
    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.special_multigammaln_out
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", MULTIGAMMALN_P_VALUES)
def test_special_multigammaln_out(shape, dtype, p):
    x = _get_multigammaln_input(shape, dtype, flag_gems.device, p)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        out_ref = torch.empty_like(ref_x, dtype=torch.float32)
        ref_out = torch.ops.aten.special_multigammaln.out(ref_x.float(), p, out=out_ref)
        out_ref = out_ref.to(dtype)
        ref_out = out_ref
    else:
        out_ref = torch.empty_like(ref_x)
        ref_out = torch.ops.aten.special_multigammaln.out(ref_x, p, out=out_ref)
    out_act = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_multigammaln.out(x, p, out=out_act)
    utils.gems_assert_close(act_out, ref_out, dtype)
    utils.gems_assert_close(out_act, out_ref, dtype)


@pytest.mark.special_multigammaln
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_special_multigammaln_int_promotes(dtype):
    x = torch.full((2, 3), 2, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.special_multigammaln(ref_x, 2)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_multigammaln(x, 2)
    assert act_out.dtype is torch.float32
    utils.gems_assert_close(act_out, ref_out, torch.float32)


@pytest.mark.special_multigammaln
def test_special_multigammaln_bool_raises():
    x = torch.ones((2, 3), dtype=torch.bool, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.special_multigammaln(x, 2)


@pytest.mark.special_multigammaln
def test_special_multigammaln_invalid_p_raises():
    x = torch.full((2, 3), 2.0, dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.special_multigammaln(x, 0)


@pytest.mark.special_multigammaln_out
def test_special_multigammaln_out_casts_and_resizes():
    x = torch.full((2, 3), 2.0, dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    out_ref = torch.empty((0,), dtype=torch.float64, device=ref_x.device)
    ref_out = torch.ops.aten.special_multigammaln.out(ref_x, 2, out=out_ref)
    out_act = torch.empty((0,), dtype=torch.float64, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_multigammaln.out(x, 2, out=out_act)
    assert act_out.data_ptr() == out_act.data_ptr()
    assert tuple(out_act.shape) == tuple(x.shape)
    utils.gems_assert_close(act_out, ref_out, torch.float64)
