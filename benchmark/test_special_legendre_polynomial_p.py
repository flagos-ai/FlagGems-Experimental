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

from . import base


class LegendrePolynomialPBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return None


def legendre_polynomial_p_input_fn(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    n = 3  # fixed polynomial degree
    n_tensor = torch.tensor(n, dtype=torch.long, device=device)
    yield x, n_tensor


@pytest.mark.special_legendre_polynomial_p
def test_special_legendre_polynomial_p():
    bench = LegendrePolynomialPBenchmark(
        input_fn=legendre_polynomial_p_input_fn,
        op_name="special_legendre_polynomial_p",
        torch_op=torch.ops.aten.special_legendre_polynomial_p,
        dtypes=[torch.float32],  # PyTorch legendre_polynomial_p only supports float32
    )
    bench.run()
