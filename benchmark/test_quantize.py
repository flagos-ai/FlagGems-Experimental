import pytest
import torch

import flag_gems

from . import base


@pytest.mark.quantize
def test_quantize():
    bench = base.UnaryPointwiseBenchmark(
        op_name="quantize",
        torch_op=lambda x: torch.clamp(torch.round(x / 0.1 + 10), 0, 255).to(x.dtype),
        gems_op=lambda x: flag_gems.quantize(x, 0.1, 10),
        dtypes=[torch.float32],
    )
    bench.run()
