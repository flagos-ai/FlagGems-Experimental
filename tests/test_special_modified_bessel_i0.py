import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_modified_bessel_i0
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_modified_bessel_i0(shape, dtype):
    inp = torch.randn(
        shape,
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    # Compute the reference result on CPU in float32.
    ref_inp_cpu = ref_inp.to("cpu").float()
    ref_out_cpu = torch.special.modified_bessel_i0(ref_inp_cpu)

    # In quick-CPU mode, the comparison utility moves the actual result
    # to CPU and expects the reference result to already be on CPU.
    reference_device = torch.device("cpu") if utils.TO_CPU else flag_gems.device
    ref_out = ref_out_cpu.to(reference_device)

    with flag_gems.use_gems():
        res_out = torch.special.modified_bessel_i0(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_modified_bessel_i0_out
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_modified_bessel_i0_out(shape, dtype):
    inp = torch.randn(
        shape,
        dtype=dtype,
        device=flag_gems.device,
    )
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)

    # Compute the reference result on CPU in float32.
    ref_inp_cpu = ref_inp.to("cpu").float()
    ref_out_cpu = torch.empty_like(ref_inp_cpu)
    torch.special.modified_bessel_i0(
        ref_inp_cpu,
        out=ref_out_cpu,
    )

    # Keep the reference result on CPU in quick-CPU mode.
    reference_device = torch.device("cpu") if utils.TO_CPU else flag_gems.device
    ref_out = ref_out_cpu.to(reference_device)

    with flag_gems.use_gems():
        torch.special.modified_bessel_i0(inp, out=out)

    utils.gems_assert_close(out, ref_out, dtype)
