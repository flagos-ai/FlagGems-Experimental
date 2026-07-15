import pytest
import torch

from flag_gems.ops.special_chebyshev_polynomial_t import special_chebyshev_polynomial_t

from . import base


class _SpecialChebyshevPolynomialTBenchmark(base.BinaryPointwiseBenchmark):
    def get_input_iter(self, dtype):
        for shape in self.shapes:
            x = base.generate_tensor_input(shape, dtype, self.device)
            n = torch.randint(-2, 21, shape, dtype=torch.int32, device=self.device)
            yield x, n


@pytest.mark.special_chebyshev_polynomial_t
def test_special_chebyshev_polynomial_t():
    bench = _SpecialChebyshevPolynomialTBenchmark(
        op_name="special_chebyshev_polynomial_t",
        torch_op=torch.special.chebyshev_polynomial_t,
        gems_op=special_chebyshev_polynomial_t,
        dtypes=[torch.float32],
    )
    bench.run()
