import pytest
import torch

from . import base


def eigvalsh_input_fn(shape, dtype, device):
    n = min(shape[-1], 4096)
    A = torch.randn(n, n, dtype=dtype, device=device)
    A = (A + A.mT) / 2
    yield A,


@pytest.mark.linalg_eigvalsh
def test_linalg_eigvalsh():
    bench = base.GenericBenchmark(
        op_name="linalg_eigvalsh",
        input_fn=eigvalsh_input_fn,
        torch_op=torch.linalg.eigvalsh,
        # torch.linalg.eigvalsh on CUDA only supports float32 and float64
        dtypes=[torch.float32],
    )
    bench.run()
