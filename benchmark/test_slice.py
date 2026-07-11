import pytest
import torch

from . import base, consts


class TensorSelectBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        if (
            base.vendor_name == "kunlunxin"
        ):  # Speed Up Benchmark Test, Big Shape Will Cause Timeout
            return []
        shapes = super().set_more_shapes()
        shapes = [
            # this filter is for scatter
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]
        return shapes


def slice_gbps(args, latency):
    inp, dim, start, end, step = args
    bytes_per_element = inp.element_size()

    total_bytes = inp.numel() * bytes_per_element

    return total_bytes / latency / 1e9


@pytest.mark.slice
def test_slice_perf():
    def slice_input_fn(shape, dtype, device):
        dim = 0 if len(shape) == 1 else 1

        start = 0
        end = shape[dim]
        step = 2

        size = shape[dim]

        start = start % size
        end = end % (size + 1)

        if end < start:
            end, start = start, end
        elif end == start:
            end = size

        inp = torch.randn(shape, dtype=dtype, device=device)

        yield inp, dim, start, end, step

    bench = TensorSelectBenchmark(
        op_name="slice",
        torch_op=torch.ops.aten.slice,
        input_fn=slice_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=slice_gbps,
    )

    bench.run()
