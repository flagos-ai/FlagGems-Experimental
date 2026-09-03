import pytest
import torch

import flag_gems


@pytest.mark.unique_consecutive
def test_unique_consecutive():
    """
    Benchmark for unique_consecutive operator.

    Note: This operator has dynamic output size, so standard benchmarking
    is not applicable. This test serves as a placeholder to document the operator.
    """
    # Simple smoke test
    inp = torch.randn(1024, dtype=torch.float32, device="cuda")
    inp = torch.round(inp * 10) / 10  # Create duplicates

    # Warm up
    for _ in range(10):
        _ = flag_gems.unique_consecutive(inp)

    # Run once to verify it works
    result = flag_gems.unique_consecutive(inp, return_inverse=True, return_counts=True)
    assert len(result) == 3
    print(
        f"unique_consecutive: input shape {inp.shape}, output shape {result[0].shape}"
    )


@pytest.mark.unique_consecutive_out
def test_unique_consecutive_out():
    """
    Benchmark for unique_consecutive out variant.

    Note: This operator has dynamic output size, so standard benchmarking
    is not applicable. This test serves as a placeholder to document the operator.
    """
    # Simple smoke test
    inp = torch.randn(1024, dtype=torch.float32, device="cuda")
    inp = torch.round(inp * 10) / 10  # Create duplicates

    out0 = torch.empty(0, dtype=torch.float32, device="cuda")
    out1 = torch.empty(0, dtype=torch.long, device="cuda")
    out2 = torch.empty(0, dtype=torch.long, device="cuda")

    # Warm up
    for _ in range(10):
        out0 = torch.empty(0, dtype=torch.float32, device="cuda")
        out1 = torch.empty(0, dtype=torch.long, device="cuda")
        out2 = torch.empty(0, dtype=torch.long, device="cuda")
        _ = flag_gems.unique_consecutive_out(
            inp, True, True, None, out0=out0, out1=out1, out2=out2
        )

    # Run once to verify it works
    out0 = torch.empty(0, dtype=torch.float32, device="cuda")
    out1 = torch.empty(0, dtype=torch.long, device="cuda")
    out2 = torch.empty(0, dtype=torch.long, device="cuda")
    result = flag_gems.unique_consecutive_out(
        inp, True, True, None, out0=out0, out1=out1, out2=out2
    )
    assert len(result) == 3
    print(
        f"unique_consecutive_out: input shape {inp.shape}, output shape {result[0].shape}"
    )
