# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from . import base, consts


class SpecialLegendrePolynomialPBenchmark(base.Benchmark):
    """Benchmark for special_legendre_polynomial_p (Legendre polynomial).

    This is a binary operation where the first input is a tensor and the
    second input is a scalar polynomial degree.
    """

    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # x is the input tensor, n is the polynomial degree (scalar)
            x = base.generate_tensor_input(shape, cur_dtype, self.device)
            n = 3
            yield x, n

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


@pytest.mark.special_legendre_polynomial_p
def test_special_legendre_polynomial_p():
    bench = SpecialLegendrePolynomialPBenchmark(
        op_name="special_legendre_polynomial_p",
        torch_op=torch.special.legendre_polynomial_p,
        # special.legendre_polynomial_p only supports float32 in PyTorch
        dtypes=[torch.float32],
    )
    bench.run()
