import pytest
import torch

from . import base, consts


class HardshrinkBenchmark(base.GenericBenchmark):
    # hardshrink takes (x, lambd) — not a simple unary op,
    # use GenericBenchmark with custom input_fn
    def set_more_shapes(self):
        return None


def hardshrink_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    # lambd=0.5 is the default in PyTorch; benchmark against the standard value
    lambd = 0.5
    yield inp, lambd


@pytest.mark.hardshrink
def test_hardshrink():
    bench = HardshrinkBenchmark(
        input_fn=hardshrink_input_fn,
        op_name="hardshrink",
        torch_op=torch.ops.aten.hardshrink,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
