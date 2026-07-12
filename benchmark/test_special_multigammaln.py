import pytest
import torch

from . import base, consts


@pytest.mark.special_multigammaln
def test_special_multigammaln():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_multigammaln",
        torch_op=lambda a: torch.ops.aten.special_multigammaln(a, 2),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
