import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_chebyshev_polynomial_u
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES + [(2, 1)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_chebyshev_polynomial_u(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if shape == (2, 1):
        n = torch.tensor([0, 1, 2], dtype=torch.long, device=flag_gems.device)
    else:
        n = torch.randint(-1, 10, shape, dtype=torch.long, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_n = utils.to_reference(n)

    # For float16/bfloat16, convert to float32 for reference computation
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_chebyshev_polynomial_u(
            ref_x.float(), ref_n
        ).to(dtype)
    else:
        ref_out = torch.ops.aten.special_chebyshev_polynomial_u(ref_x, ref_n)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_chebyshev_polynomial_u(x, n)

    # Use a larger tolerance for larger n values due to numerical instability
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.special_chebyshev_polynomial_u
def test_special_chebyshev_polynomial_u_scalar_and_promote():
    x = torch.randn((4,), dtype=torch.float32, device=flag_gems.device)
    n_scalar = torch.tensor(2, dtype=torch.long, device=flag_gems.device)
    n_promote = torch.tensor([2.0], dtype=torch.float64, device=flag_gems.device)

    ref_out = torch.ops.aten.special_chebyshev_polynomial_u(
        utils.to_reference(x), utils.to_reference(n_scalar)
    )
    ref_promote = torch.ops.aten.special_chebyshev_polynomial_u(
        utils.to_reference(x), utils.to_reference(n_promote)
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_chebyshev_polynomial_u(x, n_scalar)
        res_promote = torch.ops.aten.special_chebyshev_polynomial_u(x, n_promote)

    utils.gems_assert_close(res_out, ref_out, torch.float32, atol=1e-6)
    assert res_promote.dtype == torch.float64
    utils.gems_assert_close(res_promote, ref_promote, torch.float64, atol=1e-6)


@pytest.mark.special_chebyshev_polynomial_u
def test_special_chebyshev_polynomial_u_high_order():
    x = torch.tensor([0.5, -0.25, 1.0], dtype=torch.float32, device=flag_gems.device)
    n = torch.tensor([31, 32, 40], dtype=torch.long, device=flag_gems.device)

    ref_out = torch.ops.aten.special_chebyshev_polynomial_u(
        utils.to_reference(x), utils.to_reference(n)
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_chebyshev_polynomial_u(x, n)

    utils.gems_assert_close(res_out, ref_out, torch.float32, atol=1e-3)
