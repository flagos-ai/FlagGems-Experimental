import pytest
import torch

from . import base, utils


class SpecialBesselJ0Benchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "tflops" not in self.metrics:
            self.metrics = self.metrics[:] + ["tflops"]
            self.to_bench_metrics = self.metrics

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            if cur_dtype == torch.float64:
                inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            else:
                inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            yield inp,

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


@pytest.mark.special_bessel_j0
def test_special_bessel_j0():
    bench = SpecialBesselJ0Benchmark(
        op_name="special_bessel_j0",
        torch_op=torch.special.bessel_j0,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
