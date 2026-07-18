import pytest
import torch

from . import base


@pytest.mark.special_bessel_y0
def test_special_bessel_y0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_bessel_y0",
        torch_op=torch.special.bessel_y0,
        dtypes=[torch.float32],
    )
    bench.run()
