import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _digamma(x):
    PI = 3.141592653589793

    needs_reflect = x <= 0.0
    xp = tl.where(needs_reflect, 1.0 - x, x)

    t = tl.minimum(xp, 1.0e19)
    r = xp + 3.0
    inv_sum = (3.0 * t * t + 6.0 * t + 2.0) / (t * (t + 1.0) * (t + 2.0))

    rinv = 1.0 / r
    z = rinv * rinv
    poly = z * (
        1.0 / 12.0 + z * (-1.0 / 120.0 + z * (1.0 / 252.0 + z * (-1.0 / 240.0)))
    )
    psi = tl.math.log(r) - 0.5 * rinv - poly - inv_sum

    cot = tl.math.cos(PI * x) / tl.math.sin(PI * x)
    psi = tl.where(needs_reflect, psi - PI * cot, psi)
    return psi


@triton.jit
def digamma_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=1.0)
    y = _digamma(x.to(tl.float32))
    tl.store(out_ptr + offsets, y, mask=mask)


def digamma(input):
    logger.debug("GEMS_KUNLUNXIN DIGAMMA")
    output = torch.empty_like(input)
    n = input.numel()
    if n == 0:
        return output
    if n >= (1 << 22):
        if input.dtype == torch.bfloat16:
            BLOCK_SIZE, NUM_WARPS = 16384, 16
        else:
            BLOCK_SIZE, NUM_WARPS = 8192, 8
    else:
        BLOCK_SIZE, NUM_WARPS = 1024, 4
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    digamma_kernel[grid](input, output, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return output
