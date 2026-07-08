import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.ne
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test_ne(shape, dtype):
    if dtype in utils.ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = inp1.clone()
        if inp2.numel() > 0:
            inp2.reshape(-1)[0] = inp2.reshape(-1)[0] + 1
    elif dtype in utils.ALL_INT_DTYPES:
        inp1 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    elif dtype in utils.BOOL_TYPES:
        inp1 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.ne(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.ne(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.ne
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test_ne_scalar(shape, dtype):
    if dtype in utils.ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.ALL_INT_DTYPES:
        inp1 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    elif dtype in utils.BOOL_TYPES:
        inp1 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    inp2 = 0
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.ne(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.ne(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)
