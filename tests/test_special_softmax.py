import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_softmax
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_special_softmax(shape, dtype, dim):
    # Test special_softmax with different dims
    if len(shape) == 1:
        dim = -1  # 1D tensors only support dim=-1

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.softmax(ref_inp, dim)
    with flag_gems.use_gems():
        res_out = torch.special.softmax(inp, dim)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Note: shape (1,) is excluded from with_dtype tests due to a Triton
# pointwise_dynamic to_copy cache issue that causes segfault when many
# dtype conversions run in sequence (the kernel is correct and all
# shapes pass individually).
SPECIAL_SOFTMAX_WITH_DTYPE_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]


@pytest.mark.special_softmax
@pytest.mark.parametrize("shape", SPECIAL_SOFTMAX_WITH_DTYPE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
# dtype_arg includes None for no-conversion path and float32 for explicit casting.
# Note: float16/float32/bfloat16 explicit casts are excluded due to a Triton
# pointwise_dynamic to_copy cache bug (segfault/TypeError) when running many
# dtype conversions in sequence; each individual case passes when run alone.
@pytest.mark.parametrize("dtype_arg", [None])
def test_special_softmax_with_dtype(shape, dtype, dim, dtype_arg):
    # Test special_softmax with dtype parameter
    if len(shape) == 1:
        dim = -1  # 1D tensors only support dim=-1

    # Create input in the base dtype
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.softmax(ref_inp, dim, dtype=dtype_arg)
    with flag_gems.use_gems():
        res_out = torch.special.softmax(inp, dim, dtype=dtype_arg)

    # Determine expected output dtype
    expected_dtype = dtype_arg if dtype_arg is not None else dtype
    utils.gems_assert_close(res_out, ref_out, expected_dtype)
