import pytest
import torch

from . import base


def _sym_constrain_range_input_fn(shape, cur_dtype, device):
    # shape parameter is ignored since we use scalar inputs
    # Yield (size, min, max) tuple
    yield (5, {"min": 0, "max": 10})


@pytest.mark.sym_constrain_range
@pytest.mark.performance
def test_sym_constrain_range():
    # Use a simple shape just to satisfy the benchmark framework
    bench = base.GenericBenchmark(
        op_name="sym_constrain_range",
        torch_op=torch.sym_constrain_range,
        dtypes=[torch.float32],
        input_fn=_sym_constrain_range_input_fn,
    )
    bench.run()


@pytest.mark.sym_constrain_range_for_size
@pytest.mark.performance
def test_sym_constrain_range_for_size():
    bench = base.GenericBenchmark(
        op_name="sym_constrain_range_for_size",
        torch_op=torch.sym_constrain_range_for_size,
        dtypes=[torch.float32],
        input_fn=_sym_constrain_range_input_fn,
    )
    bench.run()
