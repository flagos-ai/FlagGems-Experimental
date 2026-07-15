import pytest
import torch

from . import base


@pytest.mark.special_modified_bessel_k0
def test_special_modified_bessel_k0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_modified_bessel_k0",
        torch_op=torch.special.modified_bessel_k0,
        # torch.special.modified_bessel_k0 only supports float32
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_modified_bessel_k0_out
def test_special_modified_bessel_k0_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_modified_bessel_k0_out",
        torch_op=torch.special.modified_bessel_k0,
        # torch.special.modified_bessel_k0 only supports float32
        dtypes=[torch.float32],
    )
    bench.run()
