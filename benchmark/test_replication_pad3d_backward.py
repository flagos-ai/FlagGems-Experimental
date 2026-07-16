import pytest
import torch

from . import base

REPLICATION_PAD3D_BACKWARD_CASES = [
    ((1, 2, 4, 5, 6), (0, 0, 0, 0, 0, 0)),
    ((2, 4, 8, 16, 16), (1, 1, 1, 1, 1, 1)),
    ((2, 4, 4, 8, 8), (2, 0, 1, 2, 0, 1)),
    ((2, 3, 4, 5), (1, 2, 0, 1, 2, 0)),
]


class ReplicationPad3dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = REPLICATION_PAD3D_BACKWARD_CASES

    def get_input_iter(self, cur_dtype):
        for shape, padding in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            out = torch.nn.functional.pad(inp, padding, mode="replicate")
            grad_output = torch.randn_like(out)
            yield grad_output, inp, padding


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward():
    bench = ReplicationPad3dBackwardBenchmark(
        op_name="replication_pad3d_backward",
        torch_op=torch.ops.aten.replication_pad3d_backward,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
