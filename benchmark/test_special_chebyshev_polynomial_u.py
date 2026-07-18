from typing import Generator

import pytest
import torch

from . import base

CHEBYSHEV_U_DTYPES = [torch.float32, torch.float64]


class ChebyshevPolynomialUBenchmark(base.BinaryPointwiseBenchmark):
    """
    Special benchmark for special_chebyshev_polynomial_u where the second input (n)
    is an integer tensor instead of float.
    """

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            # n should be a small integer tensor (0-10 for typical testing)
            inp2 = torch.randint(0, 10, shape, dtype=torch.long, device=self.device)
            yield inp1, inp2


@pytest.mark.special_chebyshev_polynomial_u
@pytest.mark.parametrize("dtype", CHEBYSHEV_U_DTYPES)
def test_special_chebyshev_polynomial_u_perf(dtype):
    bench = ChebyshevPolynomialUBenchmark(
        op_name="special_chebyshev_polynomial_u",
        torch_op=torch.ops.aten.special_chebyshev_polynomial_u,
        dtypes=[dtype],
    )
    bench.run()
