import pytest
import torch

from . import base, consts


@pytest.mark.special_expm1
def test_special_expm1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_expm1",
        torch_op=torch.special.expm1,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
