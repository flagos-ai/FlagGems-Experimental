import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_ndtri
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_ndtri(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    if inp.numel() >= 5:
        inp_flat = inp.flatten()
        inp_flat[:5] = torch.tensor(
            [0.0, 1e-6, 0.5, 1.0 - 1e-6, 1.0],
            dtype=dtype,
            device=flag_gems.device,
        )
    ref_inp = utils.to_reference(inp)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.special.ndtri(ref_inp.float()).to(dtype)
    else:
        ref_out = torch.special.ndtri(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.ndtri(inp)

    # ndtri involves inverse erf which can have higher numerical errors
    utils.gems_assert_close(res_out, ref_out, dtype, atol=5e-2)


@pytest.mark.special_ndtri
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_ndtri_out(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    if inp.numel() >= 5:
        inp_flat = inp.flatten()
        inp_flat[:5] = torch.tensor(
            [0.0, 1e-6, 0.5, 1.0 - 1e-6, 1.0],
            dtype=dtype,
            device=flag_gems.device,
        )
    ref_inp = utils.to_reference(inp)
    out_ref = torch.empty_like(ref_inp)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_ndtri.out(
            ref_inp.float(), out=torch.empty_like(ref_inp, dtype=torch.float32)
        ).to(dtype)
        out_ref.copy_(ref_out)
    else:
        ref_out = torch.ops.aten.special_ndtri.out(ref_inp, out=out_ref)
    out_act = torch.empty_like(inp)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_ndtri.out(inp, out=out_act)

    # ndtri involves inverse erf which can have higher numerical errors
    utils.gems_assert_close(act_out, ref_out, dtype, atol=5e-2)
    utils.gems_assert_close(out_act, out_ref, dtype, atol=5e-2)
