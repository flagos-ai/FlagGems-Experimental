import pytest
import torch

from . import base, consts


@pytest.mark.special_xlogy
def test_special_xlogy():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_xlogy",
        torch_op=torch.special.xlogy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.xlogy
@pytest.mark.special_xlogy
def test_xlogy():
    bench = base.BinaryPointwiseBenchmark(
        op_name="xlogy",
        torch_op=torch.special.xlogy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.xlogy_
@pytest.mark.special_xlogy
def test_xlogy_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="xlogy_",
        torch_op=lambda a, b: a.xlogy_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
