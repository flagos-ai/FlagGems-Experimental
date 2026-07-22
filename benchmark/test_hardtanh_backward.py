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


def hardtanh_backward_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    # Use default hardtanh bounds for benchmark
    min_val = -1.0
    max_val = 1.0
    yield grad_output, inp, min_val, max_val


class HardtanhBackwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return None


@pytest.mark.hardtanh_backward
def test_hardtanh_backward():
    bench = HardtanhBackwardBenchmark(
        input_fn=hardtanh_backward_input_fn,
        op_name="hardtanh_backward",
        torch_op=torch.ops.aten.hardtanh_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
