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


def shifted_chebyshev_input_fn(shape, dtype, device):
    x = torch.randn(shape, device=device, dtype=dtype)
    n = torch.randint(0, 65, shape, dtype=torch.long, device=device)
    yield x, n


@pytest.mark.special_shifted_chebyshev_polynomial_t
def test_special_shifted_chebyshev_polynomial_t():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=shifted_chebyshev_input_fn,
        op_name="special_shifted_chebyshev_polynomial_t",
        torch_op=torch.special.shifted_chebyshev_polynomial_t,
        dtypes=[torch.float32],
    )
    bench.run()
