import pytest

import flag_gems

from . import base, consts


@pytest.mark.ReduceArgMin
def test_ReduceArgMin():
    bench = base.UnaryReductionBenchmark(
        op_name="ReduceArgMin",
        torch_op=flag_gems.ReduceArgMin,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
