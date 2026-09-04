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


@pytest.mark.special_spherical_bessel_j0
def test_special_spherical_bessel_j0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_spherical_bessel_j0",
        torch_op=torch.special.spherical_bessel_j0,
        # torch.special.spherical_bessel_j0 only supports float32
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_spherical_bessel_j0_
def test_special_spherical_bessel_j0_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_spherical_bessel_j0_",
        torch_op=torch.special.spherical_bessel_j0,
        # torch.special.spherical_bessel_j0 only supports float32
        dtypes=[torch.float32],
        is_inplace=True,
    )
    bench.run()
