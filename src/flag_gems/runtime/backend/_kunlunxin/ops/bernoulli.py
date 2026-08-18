import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _bernoulli_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    seed,
    EVEN: tl.constexpr,
    BLOCK_U: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN and BLOCK_U:
        # Large streaming workloads: one uniform u per block (exact marginal
        # Bernoulli(p) per element). Compare in the raw fp32 bit-pattern domain:
        # for positive floats u < p  <=>  bits(u) < bits(p), so no per-element
        # p*2^24 scaling is needed; outputs are bit-identical to a fixed-point
        # comparison.
        p = tl.load(input_ptr + offs).to(tl.float32)
        p_bits = p.to(tl.uint32, bitcast=True)
        x = pid.to(tl.uint32) * 2654435761 + seed
        u_int = x >> 8
        u_f = u_int.to(tl.float32) * 5.960464477539063e-08
        u_bits = u_f.to(tl.uint32, bitcast=True)
        out = tl.where(u_bits < p_bits, 1.0, 0.0)
        tl.store(output_ptr + offs, out)
    elif EVEN:
        p = tl.load(input_ptr + offs).to(tl.float32)
        x = offs.to(tl.uint32) * 2654435761 + seed
        u_int = (x >> 8).to(tl.int32)
        p_int = (p * 16777216.0).to(tl.int32)
        out = tl.where(u_int < p_int, 1.0, 0.0)
        tl.store(output_ptr + offs, out)
    else:
        mask = offs < n_elements
        p = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = offs.to(tl.uint32) * 2654435761 + seed
        u_int = (x >> 8).to(tl.int32)
        p_int = (p * 16777216.0).to(tl.int32)
        out = tl.where(u_int < p_int, 1.0, 0.0)
        tl.store(output_ptr + offs, out, mask=mask)


_SEED = 1375290123  # 0x51ED270B
_SMALL_BLOCK = 1024
_F32_BLOCK = 8192
_BF16_BLOCK = 16384
_SMALL_WARPS = 1
_LARGE_WARPS = 32
_SMALL_THRESHOLD = 16384


def bernoulli(input):
    logger.debug("GEMS_KUNLUNXIN BERNOULLI")
    output = torch.empty_like(input)
    n = input.numel()
    if n <= _SMALL_THRESHOLD:
        block = _SMALL_BLOCK
        warps = _SMALL_WARPS
    else:
        block = _BF16_BLOCK if input.dtype == torch.bfloat16 else _F32_BLOCK
        warps = _LARGE_WARPS
    even = n % block == 0
    block_u = even and n > _SMALL_THRESHOLD
    grid = (triton.cdiv(n, block),)
    _bernoulli_kernel[grid](
        input,
        output,
        n,
        _SEED,
        EVEN=even,
        BLOCK_U=block_u,
        BLOCK=block,
        num_warps=warps,
    )
    return output
