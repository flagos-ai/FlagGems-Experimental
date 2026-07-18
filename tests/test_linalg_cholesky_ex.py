import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Cholesky-specific test shapes (square matrices)
CHOLESKY_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]


@pytest.mark.linalg_cholesky_ex
@pytest.mark.parametrize("shape", CHOLESKY_SHAPES)
# float16/bfloat16 not supported by cuSolver for cholesky; only float32/float64 available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_ex(shape, dtype):
    """Test Cholesky decomposition accuracy"""
    n = shape[0]

    # Create a symmetric positive definite matrix
    # A = A_orig @ A_orig.T ensures positive definiteness
    A_orig = torch.randn((n, n), dtype=dtype, device=flag_gems.device)
    A = A_orig @ A_orig.t() + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1

    ref_A = utils.to_reference(A)

    # Reference implementation
    ref_out = torch.linalg.cholesky_ex(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.cholesky_ex(A)

    # Compare L matrices
    utils.gems_assert_close(res_out.L, ref_out.L, dtype)

    # Compare info (should be 0 for positive definite matrix)
    utils.gems_assert_equal(res_out.info, ref_out.info)
