import pytest
import torch

from . import base


@pytest.mark.special_ndtri
def test_special_ndtri():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_ndtri",
        torch_op=torch.special.ndtri,
        dtypes=[torch.float32],
    )
    bench.run()
