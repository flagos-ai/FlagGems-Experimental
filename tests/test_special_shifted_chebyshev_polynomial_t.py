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

# shifted_chebyshev_polynomial_t has no eager Half/BFloat16 CUDA kernel, so the
# supported set is the float32/float64 subset of utils.FLOAT_DTYPES.
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = [
        dtype
        for dtype in (*utils.FLOAT_DTYPES, torch.float64)
        if dtype in (torch.float32, torch.float64)
    ]


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_shifted_chebyshev_polynomial_t(shape, dtype, caplog):
    # x in [0, 1] for shifted Chebyshev polynomial
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    n = torch.randint(0, 10, shape, dtype=torch.long, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_n = n.to(ref_x.device).to(ref_x.dtype)

    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(ref_x, ref_n)
    logger_name = "flag_gems.ops.special_shifted_chebyshev_polynomial_t"
    with caplog.at_level("DEBUG", logger=logger_name):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(x, n)
    assert "GEMS SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_T" in caplog.text

    # Use larger tolerance for float32 due to trigonometric function precision
    utils.gems_assert_close(res_out, ref_out, dtype, atol=5e-3)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_shifted_chebyshev_polynomial_t_scalar_n(shape, dtype, caplog):
    # Scalar n reaches the same wrapper: aten decomposes the .n_scalar overload
    # onto the registered .default overload, so no separate registration exists.
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    n = 3  # scalar

    ref_x = utils.to_reference(x, True)

    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(ref_x, n)
    logger_name = "flag_gems.ops.special_shifted_chebyshev_polynomial_t"
    with caplog.at_level("DEBUG", logger=logger_name):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(x, n)
    assert "GEMS SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_T" in caplog.text

    utils.gems_assert_close(res_out, ref_out, dtype)


# The scalar-degree path has three kernels: a compile-time unrolled one for
# ``0 <= n <= 24``, a dynamic loop above that, and a zero-result kernel for
# ``n < 0`` (the Chebyshev polynomial is undefined there and ATen returns 0).
# The cases below drive each of them; the tensor-degree path picks its branch
# per element from the same table, so those degrees are swept as tensors too.
#
# The tolerance grows with the degree, and deliberately so: the three-term
# recurrence is ill-conditioned in float32 as n grows, and both the ATen device
# kernel and this one drift from the float64 truth by the same order (measured
# at n=4096: 6.5e-3 for ATen, 5.8e-3 here). A 1e-4 bound there would test
# float32 arithmetic rather than the kernel; 2e-2 still rejects a wrong branch,
# a wrong kernel or a shape/layout error. float64 matches ATen bit-for-bit at
# every degree and keeps the default tolerance.
SCALAR_DEGREES = [0, 1, 3, 24, 25, 255, 4096, -1, -7]
HIGH_DEGREE_FP32_ATOL = 2e-2


def _atol_for(dtype, n):
    if dtype is torch.float32 and abs(int(n)) > 255:
        return HIGH_DEGREE_FP32_ATOL
    return 1e-4


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("shape", [(64,), (16, 32)])
@pytest.mark.parametrize("n", SCALAR_DEGREES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_shifted_chebyshev_polynomial_t_scalar_degrees(shape, n, dtype):
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    # The reference is always built by the shared helper, so this works under
    # both CI phases (device reference, and --ref=cpu which requires a CPU
    # reference tensor).
    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(
        utils.to_reference(x, True), n
    )
    res_out = flag_gems.special_shifted_chebyshev_polynomial_t(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=_atol_for(dtype, n))


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", [torch.float64] if utils.fp64_is_supported else [])
def test_special_shifted_chebyshev_polynomial_t_tensor_degrees(dtype):
    # One tile mixing all three kernel regimes: the per-element branch choice
    # must agree with the scalar paths. float64 only, so the recurrence's
    # float32 drift cannot mask a mis-selected branch.
    x = torch.rand((4, 8), dtype=dtype, device=flag_gems.device)
    n = torch.tensor(
        [[0, 1, 3, 24, 25, 255, 4096, -1]] * 4, dtype=torch.long, device=x.device
    )

    ref_x = utils.to_reference(x, True)
    ref_n = n.to(ref_x.device).to(ref_x.dtype)
    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(ref_x, ref_n)
    res_out = flag_gems.special_shifted_chebyshev_polynomial_t(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", [torch.float64] if utils.fp64_is_supported else [])
def test_special_shifted_chebyshev_polynomial_t_degree_broadcast(dtype):
    # A degree tensor of a different shape reaches the broadcast kernel, which
    # walks per-dimension strides instead of flat indices.
    x = torch.rand((4, 1, 8), dtype=dtype, device=flag_gems.device)
    n = torch.tensor(
        [[3, 25, -2, 0, 7, 24, 1, 4096]], dtype=torch.long, device=x.device
    )

    ref_x = utils.to_reference(x, True)
    ref_n = n.to(ref_x.device).to(ref_x.dtype)
    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(ref_x, ref_n)
    res_out = flag_gems.special_shifted_chebyshev_polynomial_t(x, n)

    assert res_out.shape == (4, 1, 8)
    utils.gems_assert_close(res_out, ref_out, dtype)
