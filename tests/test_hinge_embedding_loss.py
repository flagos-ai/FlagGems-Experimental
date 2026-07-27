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

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    # Keep quick mode small while still covering 2D tensor inputs.
    HINGE_EMBEDDING_SHAPES = [(2, 3)]
    HINGE_EMBEDDING_MARGINS = [1.0]
    HINGE_EMBEDDING_REDUCTIONS = [1]
else:
    # Cover small and large 2D shapes without making the loss test suite too slow.
    HINGE_EMBEDDING_SHAPES = [(2, 3), (128, 256), (1024, 256)]
    HINGE_EMBEDDING_MARGINS = [0.0, 0.5, 1.0]
    HINGE_EMBEDDING_REDUCTIONS = [0, 1, 2]


@pytest.mark.hinge_embedding_loss
@pytest.mark.parametrize("shape", HINGE_EMBEDDING_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("margin", HINGE_EMBEDDING_MARGINS)
@pytest.mark.parametrize("reduction", HINGE_EMBEDDING_REDUCTIONS)
def test_hinge_embedding_loss(shape, dtype, margin, reduction):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = (
        torch.randint(0, 2, shape, device=flag_gems.device, dtype=torch.int8) * 2 - 1
    ).to(dtype)

    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.hinge_embedding_loss(
        ref_inp, ref_target, margin, reduction
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.hinge_embedding_loss(inp, target, margin, reduction)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())
