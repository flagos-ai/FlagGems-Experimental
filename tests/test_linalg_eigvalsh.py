import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Square matrix shapes for eigvalsh accuracy testing
EIGVALSH_SHAPES = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
# torch.linalg.eigvalsh on CUDA only supports float32 and float64
EIGVALSH_DTYPES = [torch.float32]


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape", EIGVALSH_SHAPES)
@pytest.mark.parametrize("dtype", EIGVALSH_DTYPES)
def test_linalg_eigvalsh(shape, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = (A + A.mT) / 2
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.eigvalsh(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.eigvalsh(A)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape", EIGVALSH_SHAPES)
@pytest.mark.parametrize("dtype", EIGVALSH_DTYPES)
def test_linalg_eigvalsh_batch(shape, dtype):
    full_shape = (4,) + shape
    A = torch.randn(full_shape, dtype=dtype, device=flag_gems.device)
    A = (A + A.mT) / 2
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.eigvalsh(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.eigvalsh(A)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8)])
@pytest.mark.parametrize("dtype", EIGVALSH_DTYPES)
def test_linalg_eigvalsh_uplo(shape, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = (A + A.mT) / 2
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.eigvalsh(ref_A, UPLO="U")
    with flag_gems.use_gems():
        res_out = torch.linalg.eigvalsh(A, UPLO="U")
    utils.gems_assert_close(res_out, ref_out, dtype)
