import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_chebyshev_polynomial_t(shape, dtype):
    x = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(-1.25, 1.25)
    n = torch.randint(-2, 21, shape, dtype=torch.int32, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_n = n.to(ref_x.device)

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, ref_n)
    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_t(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_chebyshev_polynomial_t_out(dtype):
    x = torch.tensor(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    n = torch.tensor(
        [-1, 0, 1, 2, 3, 11, 20], dtype=torch.int32, device=flag_gems.device
    )

    ref_x = utils.to_reference(x, True)
    ref_n = n.to(ref_x.device)
    ref_out_buf = torch.empty_like(ref_x)
    ref_out = torch.special.chebyshev_polynomial_t(ref_x, ref_n, out=ref_out_buf)

    res_out_buf = torch.empty_like(x)
    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_t(x, n, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(res_out_buf, ref_out, dtype, equal_nan=True)
