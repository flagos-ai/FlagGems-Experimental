import pytest
import torch

from . import base

# Cholesky decomposition benchmark shapes (square matrices)
CHOLESKY_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
]


class LinalgCholeskyExBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n = shape[0]
            # Create symmetric positive definite matrix: A = A_orig @ A_orig.T + alpha*I
            A_orig = torch.randn((n, n), dtype=cur_dtype, device=self.device)
            A = (
                A_orig @ A_orig.t()
                + torch.eye(n, dtype=cur_dtype, device=self.device) * 0.1
            )
            yield (A,)


@pytest.mark.linalg_cholesky_ex
def test_linalg_cholesky_ex():
    bench = LinalgCholeskyExBenchmark(
        op_name="linalg_cholesky_ex",
        torch_op=lambda A: torch.linalg.cholesky_ex(A)[0],  # Extract L only
        # float16/bfloat16 not supported by cuSolver for cholesky; only float32 available
        dtypes=[torch.float32],
    )
    bench.run()
