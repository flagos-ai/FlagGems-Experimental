import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

TRANSPOSE_SHAPES = [(2, 3), (3, 4), (2, 3, 4), (2, 4, 8), (2, 3, 4, 5)]


@pytest.mark.transpose_
@pytest.mark.parametrize("shape", TRANSPOSE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_transpose_(shape, dtype):
    """Test transpose_ with various shapes and dimensions"""
    # Test transpose of first two dimensions
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    # Use torch.transpose (non-in-place) as reference
    ref_out = torch.transpose(ref_inp, 0, 1)
    # Save a copy of reference output for comparison
    ref_out_copy = ref_out.clone()

    with flag_gems.use_gems():
        res_out = torch.ops.aten.transpose_(inp, 0, 1)

    # Check that shape is correctly modified
    assert (
        res_out.shape == ref_out_copy.shape
    ), f"Shape mismatch: {res_out.shape} vs {ref_out_copy.shape}"
    # Check that stride is correctly modified
    assert (
        res_out.stride() == ref_out_copy.stride()
    ), f"Stride mismatch: {res_out.stride()} vs {ref_out_copy.stride()}"
    # Check that data pointer is unchanged (in-place)
    assert res_out.data_ptr() == inp.data_ptr(), "transpose_ should be in-place"
    # Check data equality
    utils.gems_assert_equal(res_out, ref_out_copy)


@pytest.mark.transpose_
@pytest.mark.parametrize("shape", TRANSPOSE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_transpose_dim_middle(shape, dtype):
    """Test transpose_ with middle dimensions (for tensors with >=3 dims)"""
    if len(shape) < 3:
        pytest.skip("Need at least 3 dimensions")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    # Test transpose of dimensions 1 and 2
    ref_out = torch.transpose(ref_inp, 1, 2)
    ref_out_copy = ref_out.clone()

    with flag_gems.use_gems():
        res_out = torch.ops.aten.transpose_(inp, 1, 2)

    assert res_out.shape == ref_out_copy.shape
    assert res_out.stride() == ref_out_copy.stride()
    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out_copy)


@pytest.mark.transpose_
@pytest.mark.parametrize("shape", TRANSPOSE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_transpose_negative_dim(shape, dtype):
    """Test transpose_ with negative dimensions"""
    if len(shape) < 2:
        pytest.skip("Need at least 2 dimensions")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    # Test transpose with negative dimension (-1)
    ref_out = torch.transpose(ref_inp, 0, -1)
    ref_out_copy = ref_out.clone()

    with flag_gems.use_gems():
        res_out = torch.ops.aten.transpose_(inp, 0, -1)

    assert res_out.shape == ref_out_copy.shape
    assert res_out.stride() == ref_out_copy.stride()
    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out_copy)


@pytest.mark.transpose_
@pytest.mark.parametrize("shape", [(2, 3)])
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_transpose_int(shape, dtype):
    """Test transpose_ with integer dtypes"""
    inp = torch.randint(-10, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.transpose(ref_inp, 0, 1)
    ref_out_copy = ref_out.clone()

    with flag_gems.use_gems():
        res_out = torch.ops.aten.transpose_(inp, 0, 1)

    assert res_out.shape == ref_out_copy.shape
    assert res_out.stride() == ref_out_copy.stride()
    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out_copy)


@pytest.mark.transpose_
@pytest.mark.parametrize(
    "shape, dim0, dim1", [((), 0, 0), ((), -1, -1), ((2, 3), 1, 1)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_transpose_same_dim(shape, dim0, dim1, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.transpose_(ref_inp, dim0, dim1)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.transpose_(inp, dim0, dim1)

    assert res_out is inp
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.transpose_
@pytest.mark.parametrize(
    "shape, dim0, dim1", [((2, 3), 2, 0), ((2, 3), -3, 0), ((), 0, 1)]
)
def test_transpose_invalid_dim(shape, dim0, dim1):
    inp = torch.randn(shape, device=flag_gems.device)

    with flag_gems.use_gems():
        with pytest.raises(IndexError, match="Dimension out of range"):
            torch.ops.aten.transpose_(inp, dim0, dim1)
