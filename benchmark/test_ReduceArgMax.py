import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.ReduceArgMax
def test_ReduceArgMax():
    bench = base.UnaryReductionBenchmark(
        op_name="ReduceArgMax",
        torch_op=torch.argmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.ReduceArgMax)
    bench.run()
