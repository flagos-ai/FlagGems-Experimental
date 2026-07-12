import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.xlogy
@pytest.mark.special_xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.special.xlogy(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.special.xlogy(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.xlogy_
@pytest.mark.special_xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.special.xlogy(ref_inp1, ref_inp2, out=ref_inp1)
    with flag_gems.use_gems():
        res_out = torch.special.xlogy(inp1, inp2, out=inp1)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    assert (
        res_out is inp1
    ), "xlogy_ should return the input tensor itself to support chaining"


@pytest.mark.xlogy
@pytest.mark.special_xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_tensor_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = 2.0
    ref_inp1 = utils.to_reference(inp1, True)

    ref_out = torch.special.xlogy(ref_inp1, scalar)
    with flag_gems.use_gems():
        res_out = torch.special.xlogy(inp1, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.xlogy
@pytest.mark.special_xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_scalar_tensor(shape, dtype):
    scalar = 2.0
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.special.xlogy(scalar, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.special.xlogy(scalar, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.xlogy
@pytest.mark.special_xlogy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_edge_values(dtype):
    inp1 = torch.tensor(
        [0.0, 0.0, 0.0, 2.0, -2.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp2 = torch.tensor(
        [-1.0, 0.0, float("nan"), -1.0, 0.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.special.xlogy(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.special.xlogy(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.xlogy
@pytest.mark.special_xlogy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_scalar_tensor_edge_values(dtype):
    inp = torch.tensor(
        [-1.0, 0.0, float("nan"), 0.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.xlogy(0.0, ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.xlogy(0.0, inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
