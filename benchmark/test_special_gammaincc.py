import pytest
import torch

from . import base


@pytest.mark.special_gammaincc
def test_special_gammaincc():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_gammaincc",
        torch_op=torch.special.gammaincc,
        # Special operations
        dtypes=[torch.float32],
    )
    bench.run()
