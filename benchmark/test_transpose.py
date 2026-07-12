from typing import Generator

import pytest
import torch

from . import base, consts, utils


class TransposeBenchmark(base.Benchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            if len(shape) >= 2:
                inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
                yield inp, 0, 1  # dim0, dim1


@pytest.mark.transpose_
def test_transpose_():
    bench = TransposeBenchmark(
        op_name="transpose_",
        torch_op=torch.ops.aten.transpose_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
