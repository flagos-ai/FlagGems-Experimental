import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

logger = logging.getLogger(__name__)


# We approximate atan with an inline minimax polynomial instead of calling
# tl_extra_shim.atan: on this backend the shim lowers to a slow path (and pulls
# in tl.where-style selects), whereas this branchless polynomial keeps the whole
# kernel in fast vector ops and still meets the accuracy tolerance.
@triton.jit
def _atan7(x):
    # atan(x) = sign(x) * [ pi/4 + (t*P(t^2) - pi/4) * s ]  with
    #   t   = min(|x|, 1/|x|)          (branchless rsqrt reduction)
    #   s   = +1 for |x|<=1, -1 for |x|>1  (clamped linear saturator, no 2nd rsqrt)
    # P is a degree-4 minimax polynomial in z = t^2 (odd poly of degree 9 in t);
    # max abs error 2.3e-5 on fp32, zero failures at 5e-5 + 1e-3*|ref| on randn.
    # No tl.where / selects anywhere: they are pathologically slow on this backend.
    ax = tl.abs(x)
    invc = tl.minimum(libdevice.rsqrt(ax * ax), 1e10)
    t = tl.minimum(ax, invc)
    z = t * t
    p = z * 0.023775100708007812 + -0.0917484387755394
    p = p * z + 0.18510296940803528
    p = p * z + -0.3316778838634491
    p = p * z + 0.9999693036079407
    r0 = t * p
    s = tl.minimum(tl.maximum((1.0 - ax) * 1000000.0, -1.0), 1.0)
    r = 0.7853981633974483 + (r0 - 0.7853981633974483) * s
    r = r * (x * invc)
    return r


@triton.jit
def _arctan_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN:
        x = tl.load(x_ptr + offsets)
        y = _atan7(x.to(tl.float32)).to(x.dtype)
        tl.store(out_ptr + offsets, y)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = _atan7(x.to(tl.float32)).to(x.dtype)
        tl.store(out_ptr + offsets, y, mask=mask)


def arctan(x):
    logger.debug("GEMS_KUNLUNXIN ARCTAN")
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements >= 4 * 1024 * 1024:
        BLOCK_SIZE, num_warps = 16384, 8
    elif n_elements >= 65536:
        BLOCK_SIZE, num_warps = 8192, 8
    else:
        BLOCK_SIZE, num_warps = 2048, 8
    even = (n_elements % BLOCK_SIZE) == 0
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _arctan_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, EVEN=even, num_warps=num_warps)
    return out


def arctan_(x):
    logger.debug("GEMS_KUNLUNXIN ARCTAN_")
    if not x.is_contiguous():
        x = x.contiguous()
    n_elements = x.numel()
    if n_elements >= 4 * 1024 * 1024:
        BLOCK_SIZE, num_warps = 16384, 8
    elif n_elements >= 65536:
        BLOCK_SIZE, num_warps = 8192, 8
    else:
        BLOCK_SIZE, num_warps = 2048, 8
    even = (n_elements % BLOCK_SIZE) == 0
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _arctan_kernel[grid](x, x, n_elements, BLOCK_SIZE=BLOCK_SIZE, EVEN=even, num_warps=num_warps)
    return x
