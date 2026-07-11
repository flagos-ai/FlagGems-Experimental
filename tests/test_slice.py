import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.slice
@pytest.mark.parametrize("shape", [(16, 32, 64), (8, 16, 32), (4, 8, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("start", [0, 4, 8])
@pytest.mark.parametrize("end", [8, 12, 16])
@pytest.mark.parametrize("step", [1, 2])
def test_slice(shape, dtype, start, end, step):
    # Test slicing on dim 1 by default
    dim = 1
    dim_size = shape[dim]

    start = start % dim_size
    end = end % (dim_size + 1)

    if end < start:
        end, start = start, end
    elif end == start:
        end = dim_size

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.slice(ref_inp, dim, start, end, step)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.slice.Tensor(inp, dim, start, end, step)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice
@pytest.mark.parametrize("shape", [(16, 32, 64), (8, 16, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_dim0(shape, dtype):
    dim = 0
    start = 1
    end = shape[dim] - 1

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.slice(ref_inp, dim, start, end)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.slice.Tensor(inp, dim, start, end)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice
@pytest.mark.parametrize("shape", [(16, 32, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_negative_indices(shape, dtype):
    # Test negative start/end indices
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.slice(ref_inp, 1, -8, -2)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.slice.Tensor(inp, 1, -8, -2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice
def test_slice_preserves_view_layout():
    base = torch.arange(4 * 6 * 5, dtype=torch.float32, device=flag_gems.device)
    inp = base.reshape(4, 6, 5).transpose(0, 1)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.slice.Tensor(ref_inp, 1, 1, 4, 2)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.slice.Tensor(inp, 1, 1, 4, 2)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
