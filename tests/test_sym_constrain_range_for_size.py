import pytest
import torch

import flag_gems


@pytest.mark.sym_constrain_range_for_size
@pytest.mark.parametrize(
    "size, min_val, max_val",
    [
        (0, None, None),
        (1, None, 10),
        (5, 0, 10),
        (10, 0, 10),
        (100, None, 1000),
        (1024, 0, None),
    ],
)
def test_sym_constrain_range_for_size(size, min_val, max_val):
    ref_result = torch.sym_constrain_range_for_size(size, min=min_val, max=max_val)
    with flag_gems.use_gems():
        gems_result = torch.sym_constrain_range_for_size(size, min=min_val, max=max_val)

    assert ref_result is None
    assert gems_result is None


@pytest.mark.sym_constrain_range_for_size
@pytest.mark.parametrize(
    "size, min_val, max_val",
    [
        (-1, None, None),
        (5, 6, 10),
        (5, 0, 4),
        (1, 10, 0),
    ],
)
def test_sym_constrain_range_for_size_error(size, min_val, max_val):
    with pytest.raises(RuntimeError, match="Invalid value range|Max value"):
        torch.sym_constrain_range_for_size(size, min=min_val, max=max_val)
    with flag_gems.use_gems():
        with pytest.raises(RuntimeError, match="Invalid value range|Max value"):
            torch.sym_constrain_range_for_size(size, min=min_val, max=max_val)
