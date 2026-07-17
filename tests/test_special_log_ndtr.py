import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Common edge values for pointwise operator testing
_POINTWISE_EDGE_VALUES = [
    float("inf"), float("-inf"), float("nan"), 0.0, -0.0,
    3.4028235e38, 1e10, 1e6, 1e-6, 1e-10,
    -10.0, -5.0, -3.0, -2.0, -1.0, -0.5,
    0.5, 1.0, 2.0, 3.0, 5.0, 10.0,
    -1e-7, 1e-7,
]

# Values where erf(x/sqrt(2)) is so close to -1 that 1+erf underflows in float32
_LOG_NDTR_DANGER = [-6.5, -6.0, -5.5, -5.0, -37.0, -20.0, -15.0, 37.0]


@pytest.mark.special_log_ndtr
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_log_ndtr(shape, dtype):
    torch.manual_seed(0)
    # Use sigma=10 so ~27% of samples naturally fall in the x < -6 region
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 10.0

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.ops.aten.special_log_ndtr(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_log_ndtr(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.special_log_ndtr
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_log_ndtr_edge(dtype):
    all_vals = _POINTWISE_EDGE_VALUES + _LOG_NDTR_DANGER
    inp = torch.tensor(all_vals, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.ops.aten.special_log_ndtr(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_log_ndtr(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=1e-3)
