import pytest
import torch

from . import base


class SpecialNdtriBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.rand(shape, dtype=cur_dtype, device=self.device)
            yield inp,


@pytest.mark.special_ndtri
def test_special_ndtri():
    bench = SpecialNdtriBenchmark(
        op_name="special_ndtri",
        torch_op=torch.special.ndtri,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
