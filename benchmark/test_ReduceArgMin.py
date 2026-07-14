import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.ReduceArgMin
def test_reduce_arg_min():
    bench = base.UnaryReductionBenchmark(
        op_name="ReduceArgMin",
        torch_op=torch.argmin,
        gems_op=flag_gems.ReduceArgMin,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
