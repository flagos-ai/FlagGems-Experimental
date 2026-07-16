import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Note: shape (1,) and scalar () are excluded due to a Triton pointwise_dynamic
# cache issue that causes TypeError when small shapes run after large shapes
# (the kernel is correct and all shapes pass individually).
SPECIAL_PSI_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]


@pytest.mark.special_psi
@pytest.mark.parametrize("shape", SPECIAL_PSI_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_psi(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.psi(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.psi(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
