import pytest
import torch

import flag_gems


@pytest.mark.sym_constrain_range
@pytest.mark.parametrize(
    "size, min_val, max_val",
    [
        (5, None, None),
        (5, 0, 10),
        (5, 5, 5),
        (0, 0, 10),
        (10, 0, 10),
    ],
)
def test_sym_constrain_range(size, min_val, max_val):
    ref_out = torch.sym_constrain_range(size, min=min_val, max=max_val)
    with flag_gems.use_gems():
        res_out = torch.sym_constrain_range(size, min=min_val, max=max_val)

    assert res_out is None
    assert ref_out is None


@pytest.mark.sym_constrain_range
@pytest.mark.parametrize(
    "size, min_val, max_val",
    [
        (5, 6, None),
        (5, None, 4),
        (5, 10, 0),
        (-1, 0, 10),
        (11, 0, 10),
    ],
)
def test_exception_sym_constrain_range(size, min_val, max_val):
    with pytest.raises(RuntimeError):
        torch.sym_constrain_range(size, min=min_val, max=max_val)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.sym_constrain_range(size, min=min_val, max=max_val)


@pytest.mark.sym_constrain_range_for_size
@pytest.mark.parametrize(
    "size, min_val, max_val",
    [
        (5, None, None),
        (5, 0, 10),
        (5, 5, 5),
        (0, 0, 10),
        (10, 0, 10),
    ],
)
def test_sym_constrain_range_for_size(size, min_val, max_val):
    ref_out = torch.sym_constrain_range_for_size(size, min=min_val, max=max_val)
    with flag_gems.use_gems():
        res_out = torch.sym_constrain_range_for_size(size, min=min_val, max=max_val)

    assert res_out is None
    assert ref_out is None


@pytest.mark.sym_constrain_range_for_size
@pytest.mark.parametrize(
    "size, min_val, max_val",
    [
        (5, 6, None),
        (5, None, 4),
        (5, 10, 0),
        (-1, 0, 10),
        (11, 0, 10),
    ],
)
def test_exception_sym_constrain_range_for_size(size, min_val, max_val):
    with pytest.raises(RuntimeError):
        torch.sym_constrain_range_for_size(size, min=min_val, max=max_val)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.sym_constrain_range_for_size(size, min=min_val, max=max_val)
