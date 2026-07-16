import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

REPLICATION_PAD3D_BACKWARD_CASES = [
    ((1, 2, 4, 5, 6), (0, 0, 0, 0, 0, 0)),
    ((2, 4, 8, 16, 16), (1, 1, 1, 1, 1, 1)),
    ((2, 4, 4, 8, 8), (2, 0, 1, 2, 0, 1)),
    ((2, 3, 4, 5), (1, 2, 0, 1, 2, 0)),
]


@pytest.mark.replication_pad3d_backward
@pytest.mark.parametrize("shape, padding", REPLICATION_PAD3D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad3d_backward(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    res_out = torch.nn.functional.pad(inp, padding, mode="replicate")

    grad_output = torch.randint(
        -2, 3, res_out.shape, device=flag_gems.device, dtype=torch.int32
    ).to(dtype)
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.replication_pad3d_backward(
        ref_grad_output, ref_inp, padding
    )
    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.replication_pad3d_backward(
            grad_output, inp, padding
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
