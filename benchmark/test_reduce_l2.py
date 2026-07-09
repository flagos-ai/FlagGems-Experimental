import pytest
import torch

import flag_gems

from . import base, consts, utils


def reduce_l2_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp,


@pytest.mark.reduce_l2
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_reduce_l2(dtype):
    bench = base.GenericBenchmark(
        input_fn=reduce_l2_input_fn,
        op_name="reduce_l2",
        torch_op=torch.linalg.vector_norm,
        dtypes=[dtype],
    )
    bench.set_gems(flag_gems.reduce_l2)
    bench.run()
