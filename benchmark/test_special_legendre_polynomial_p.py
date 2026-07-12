import pytest
import torch

from . import base


class LegendrePolynomialPBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return None


def legendre_polynomial_p_input_fn(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    n = 3  # fixed polynomial degree
    n_tensor = torch.tensor(n, dtype=torch.long, device=device)
    yield x, n_tensor


@pytest.mark.special_legendre_polynomial_p
def test_special_legendre_polynomial_p():
    bench = LegendrePolynomialPBenchmark(
        input_fn=legendre_polynomial_p_input_fn,
        op_name="special_legendre_polynomial_p",
        torch_op=torch.ops.aten.special_legendre_polynomial_p,
        dtypes=[torch.float32],  # PyTorch legendre_polynomial_p only supports float32
    )
    bench.run()
