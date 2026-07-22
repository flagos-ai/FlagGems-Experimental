import pytest
import torch

from . import base


def sym_constrain_range_for_size_input_fn(shape, cur_dtype, device):
    yield 5, {"min": 0, "max": 10}


@pytest.mark.sym_constrain_range_for_size
@pytest.mark.performance
def test_sym_constrain_range_for_size():
    bench = base.GenericBenchmark(
        op_name="sym_constrain_range_for_size",
        torch_op=torch.ops.aten.sym_constrain_range_for_size,
        dtypes=[torch.float32],
        input_fn=sym_constrain_range_for_size_input_fn,
    )
    bench.run()
