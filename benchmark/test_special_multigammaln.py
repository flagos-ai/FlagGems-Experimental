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


@pytest.mark.special_multigammaln_out
def test_special_multigammaln_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_multigammaln_out",
        torch_op=lambda a, out: torch.ops.aten.special_multigammaln.out(a, 2, out=out),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
