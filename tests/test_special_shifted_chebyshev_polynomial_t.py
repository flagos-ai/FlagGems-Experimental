import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_shifted_chebyshev_polynomial_t(shape, dtype):
    inp1 = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(-2, 65, shape, dtype=torch.long, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(ref_inp1, ref_inp2)

    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_t(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_shifted_chebyshev_polynomial_t_out(shape, dtype):
    inp1 = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(-2, 65, shape, dtype=torch.long, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)
    ref_out = torch.empty_like(ref_inp1)
    res_out = torch.empty_like(inp1)

    torch.ops.aten.special_shifted_chebyshev_polynomial_t.out(
        ref_inp1, ref_inp2, out=ref_out
    )

    with flag_gems.use_gems():
        returned = torch.ops.aten.special_shifted_chebyshev_polynomial_t.out(
            inp1, inp2, out=res_out
        )

    assert returned is res_out
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_shifted_chebyshev_polynomial_t_scalar_overloads(dtype):
    inp = torch.rand((16,), dtype=dtype, device=flag_gems.device)
    n = torch.randint(-2, 65, (16,), dtype=torch.long, device=flag_gems.device)
    scalar_x = 0.25
    scalar_n = 17

    ref_inp = utils.to_reference(inp)
    ref_n = utils.to_reference(n)

    ref_x_scalar = torch.ops.aten.special_shifted_chebyshev_polynomial_t.x_scalar(
        scalar_x, ref_n
    )
    ref_n_scalar = torch.ops.aten.special_shifted_chebyshev_polynomial_t.n_scalar(
        ref_inp, scalar_n
    )

    with flag_gems.use_gems():
        res_x_scalar = torch.ops.aten.special_shifted_chebyshev_polynomial_t.x_scalar(
            scalar_x, n
        )
        res_n_scalar = torch.ops.aten.special_shifted_chebyshev_polynomial_t.n_scalar(
            inp, scalar_n
        )

    utils.gems_assert_close(res_x_scalar, ref_x_scalar, ref_x_scalar.dtype)
    utils.gems_assert_close(res_n_scalar, ref_n_scalar, dtype)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_shifted_chebyshev_polynomial_t_scalar_out_overloads(dtype):
    inp = torch.rand((16,), dtype=dtype, device=flag_gems.device)
    n = torch.randint(-2, 65, (16,), dtype=torch.long, device=flag_gems.device)
    scalar_x = 0.25
    scalar_n = 17

    ref_inp = utils.to_reference(inp)
    ref_n = utils.to_reference(n)
    ref_x_scalar = torch.empty_like(ref_inp)
    ref_n_scalar = torch.empty_like(ref_inp)
    res_x_scalar = torch.empty_like(inp)
    res_n_scalar = torch.empty_like(inp)

    torch.ops.aten.special_shifted_chebyshev_polynomial_t.x_scalar_out(
        scalar_x, ref_n, out=ref_x_scalar
    )
    torch.ops.aten.special_shifted_chebyshev_polynomial_t.n_scalar_out(
        ref_inp, scalar_n, out=ref_n_scalar
    )

    with flag_gems.use_gems():
        returned_x = torch.ops.aten.special_shifted_chebyshev_polynomial_t.x_scalar_out(
            scalar_x, n, out=res_x_scalar
        )
        returned_n = torch.ops.aten.special_shifted_chebyshev_polynomial_t.n_scalar_out(
            inp, scalar_n, out=res_n_scalar
        )

    assert returned_x is res_x_scalar
    assert returned_n is res_n_scalar
    utils.gems_assert_close(res_x_scalar, ref_x_scalar, dtype)
    utils.gems_assert_close(res_n_scalar, ref_n_scalar, dtype)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_special_shifted_chebyshev_polynomial_t_unsupported_dtype(dtype):
    inp = torch.randn((16,), dtype=dtype, device=flag_gems.device)
    n = torch.randint(0, 5, (16,), dtype=torch.long, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.special.shifted_chebyshev_polynomial_t(inp, n)


@pytest.mark.special_shifted_chebyshev_polynomial_t
def test_special_shifted_chebyshev_polynomial_t_unsupported_degree():
    inp = torch.randn((16,), dtype=torch.float32, device=flag_gems.device)
    n = torch.full((16,), 65, dtype=torch.long, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(NotImplementedError):
        torch.special.shifted_chebyshev_polynomial_t(inp, n)
