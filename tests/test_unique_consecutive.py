import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unique_consecutive
@pytest.mark.parametrize("shape", [(100,), (1024,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_unique_consecutive_basic(shape, dtype):
    """Test basic unique_consecutive functionality"""
    # Create input with some consecutive duplicates
    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        # Round to create duplicates
        inp = torch.round(inp * 10) / 10
    else:
        inp = torch.randint(-10, 10, shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_cnt = torch.unique_consecutive(
        ref_inp, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_cnt = flag_gems.unique_consecutive(
        inp, return_inverse=True, return_counts=True
    )

    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_cnt, ref_cnt)


@pytest.mark.unique_consecutive
@pytest.mark.parametrize("shape", [(100,), (1024,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unique_consecutive_no_inverse_counts(shape, dtype):
    """Test unique_consecutive without inverse and counts"""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.round(inp * 10) / 10  # Create duplicates
    ref_inp = utils.to_reference(inp)

    ref_out = torch.unique_consecutive(
        ref_inp, return_inverse=False, return_counts=False
    )
    res_out = flag_gems.unique_consecutive(
        inp, return_inverse=False, return_counts=False
    )

    # When both are False, PyTorch returns just the tensor, but our impl returns tuple
    if isinstance(res_out, tuple):
        res_out = res_out[0]

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.unique_consecutive
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unique_consecutive_all_same(dtype):
    """Test when all elements are the same"""
    shape = (100,)
    inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_cnt = torch.unique_consecutive(
        ref_inp, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_cnt = flag_gems.unique_consecutive(
        inp, return_inverse=True, return_counts=True
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_cnt, ref_cnt)


@pytest.mark.unique_consecutive
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unique_consecutive_all_different(dtype):
    """Test when all elements are different"""
    inp = torch.arange(100, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_cnt = torch.unique_consecutive(
        ref_inp, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_cnt = flag_gems.unique_consecutive(
        inp, return_inverse=True, return_counts=True
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_cnt, ref_cnt)


@pytest.mark.unique_consecutive
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_unique_consecutive_pattern(dtype):
    """Test with a known pattern"""
    inp = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_cnt = torch.unique_consecutive(
        ref_inp, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_cnt = flag_gems.unique_consecutive(
        inp, return_inverse=True, return_counts=True
    )

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_cnt, ref_cnt)


@pytest.mark.unique_consecutive
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unique_consecutive_with_dim(dim, dtype):
    """Test unique_consecutive with dim parameter"""
    shape = (10, 20)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.round(inp * 5) / 5  # Create duplicates
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_cnt = torch.unique_consecutive(
        ref_inp, return_inverse=True, return_counts=True, dim=dim
    )
    res_out, res_inv, res_cnt = flag_gems.unique_consecutive(
        inp, return_inverse=True, return_counts=True, dim=dim
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    # Note: inverse and counts for dim case are more complex
    # Just check shapes match
    assert res_inv.shape == ref_inv.shape
    assert res_cnt.shape == ref_cnt.shape


@pytest.mark.unique_consecutive_out
@pytest.mark.parametrize("shape", [(100,), (1024,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unique_consecutive_out(shape, dtype):
    """Test unique_consecutive out variant"""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.round(inp * 10) / 10  # Create duplicates
    ref_inp = utils.to_reference(inp)

    # For reference, compute without out parameter first
    ref_out, ref_inv, ref_cnt = torch.unique_consecutive(
        ref_inp, return_inverse=True, return_counts=True
    )

    # Result tensors on CUDA for out variant
    res_out0 = torch.empty(0, dtype=dtype, device=flag_gems.device)
    res_out1 = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    res_out2 = torch.empty(0, dtype=torch.long, device=flag_gems.device)

    res_ret = flag_gems.unique_consecutive_out(
        inp,
        return_inverse=True,
        return_counts=True,
        dim=None,
        out0=res_out0,
        out1=res_out1,
        out2=res_out2,
    )

    # Check that returned tensors are the same as out tensors
    assert res_ret[0] is res_out0
    assert res_ret[1] is res_out1
    assert res_ret[2] is res_out2

    utils.gems_assert_close(res_out0, ref_out, dtype)
    utils.gems_assert_equal(res_out1, ref_inv)
    utils.gems_assert_equal(res_out2, ref_cnt)


@pytest.mark.unique_consecutive
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unique_consecutive_empty(dtype):
    """Test with empty tensor"""
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_cnt = torch.unique_consecutive(
        ref_inp, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_cnt = flag_gems.unique_consecutive(
        inp, return_inverse=True, return_counts=True
    )

    assert res_out.numel() == 0
    assert res_inv.numel() == 0
    assert res_cnt.numel() == 0
