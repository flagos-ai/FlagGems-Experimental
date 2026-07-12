import pytest
import torch

from . import base


@pytest.mark.special_chebyshev_polynomial_t
def test_special_chebyshev_polynomial_t():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_chebyshev_polynomial_t",
        torch_op=torch.special.chebyshev_polynomial_t,
        dtypes=[torch.float32],
    )
    bench.run()
