import pytest
import torch

from . import base


def shifted_chebyshev_input_fn(shape, dtype, device):
    x = torch.randn(shape, device=device, dtype=dtype)
    n = torch.randint(0, 65, shape, dtype=torch.long, device=device)
    yield x, n


@pytest.mark.special_shifted_chebyshev_polynomial_t
def test_special_shifted_chebyshev_polynomial_t():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=shifted_chebyshev_input_fn,
        op_name="special_shifted_chebyshev_polynomial_t",
        torch_op=torch.special.shifted_chebyshev_polynomial_t,
        dtypes=[torch.float32],
    )
    bench.run()
