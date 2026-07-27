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


def heaviside_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    values = torch.randn(shape, dtype=dtype, device=device)
    yield inp, values


@pytest.mark.heaviside
def test_heaviside():
    bench = base.GenericBenchmark(
        op_name="heaviside",
        torch_op=torch.heaviside,
        input_fn=heaviside_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def heaviside_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    values = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp, values, {"out": out}


@pytest.mark.heaviside_out
def test_heaviside_out():
    bench = base.GenericBenchmark(
        op_name="heaviside_out",
        torch_op=torch.heaviside,
        input_fn=heaviside_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
