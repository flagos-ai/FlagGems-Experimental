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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _parse_pool3d_params(kernel_size, stride, padding, dilation):
    """Parse and validate 3-D pooling parameters.

    Each parameter can be an int (applied to all 3 spatial dims) or a
    3-element tuple/list (D, H, W).
    """

    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return tuple(param)
        raise ValueError(f"Invalid {name}: {param}")

    kd, kh, kw = _parse_param(kernel_size, "kernel_size")
    sd, sh, sw = _parse_param(stride, "stride", default=(kd, kh, kw))
    pd, ph, pw = _parse_param(padding, "padding", default=(0, 0, 0))
    dd, dh, dw = _parse_param(dilation, "dilation", default=(1, 1, 1))

    if sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError(f"stride must be positive, but got stride=({sd}, {sh}, {sw})")
    if pd < 0 or ph < 0 or pw < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({pd}, {ph}, {pw})"
        )
    if dd <= 0 or dh <= 0 or dw <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dd}, {dh}, {dw})"
        )

    return kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw


@triton.jit
def max_pool3d_kernel(
    x_ptr,
    out_ptr,
    indices_ptr,
    N,
    C,
    D,
    H,
    W,
    D_out,
    H_out,
    W_out,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    total = N * C * D_out * H_out * W_out
    mask = offsets < total

    w_out = offsets % W_out
    tmp1 = offsets // W_out
    h_out = tmp1 % H_out
    tmp2 = tmp1 // H_out
    d_out = tmp2 % D_out
    tmp3 = tmp2 // D_out
    c = tmp3 % C
    n = tmp3 // C

    d_in_start = d_out * stride - padding
    h_in_start = h_out * stride - padding
    w_in_start = w_out * stride - padding

    NEG_INF: tl.constexpr = -1.0e30
    max_val = tl.full((BLOCK_SIZE,), NEG_INF, dtype=tl.float32)
    max_idx = tl.full((BLOCK_SIZE,), -1, dtype=tl.int32)

    D_spatial = D * H * W
    HW = H * W

    for kd in tl.static_range(kernel_size):
        d_in = d_in_start + kd * dilation
        d_valid = (d_in >= 0) & (d_in < D)
        for kh in tl.static_range(kernel_size):
            h_in = h_in_start + kh * dilation
            dh_valid = d_valid & (h_in >= 0) & (h_in < H)
            for kw in tl.static_range(kernel_size):
                w_in = w_in_start + kw * dilation
                valid = dh_valid & (w_in >= 0) & (w_in < W)

                flat_in = (
                    n * C * D_spatial + c * D_spatial + d_in * HW + h_in * W + w_in
                )
                spatial_idx = d_in * HW + h_in * W + w_in

                val = tl.load(
                    x_ptr + flat_in,
                    mask=mask & valid,
                    other=NEG_INF,
                )

                is_better = val > max_val
                max_val = tl.where(is_better & valid, val, max_val)
                max_idx = tl.where(is_better & valid, spatial_idx, max_idx)

    flat_out = (
        n * C * D_out * H_out * W_out
        + c * D_out * H_out * W_out
        + d_out * H_out * W_out
        + h_out * W_out
        + w_out
    )
    tl.store(out_ptr + flat_out, max_val, mask=mask)
    tl.store(indices_ptr + flat_out, max_idx, mask=mask)


def max_pool3d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Compute 3-D max pooling, returning (output, indices).

    Indices are flat offsets into the (D, H, W) spatial volume of the input.
    """
    logger.debug("GEMS_ASCEND MAX_POOL3D_WITH_INDICES_FORWARD")

    input = input.contiguous()

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params

    # This kernel requires uniform pooling parameters across D, H, W
    assert kd == kh == kw, "kernel_size must be uniform across D, H, W"
    assert sd == sh == sw, "stride must be uniform across D, H, W"
    assert pd == ph == pw, "padding must be uniform across D, H, W"
    assert dd == dh == dw, "dilation must be uniform across D, H, W"

    N, C, D, H, W = input.shape

    numerator_d = D + 2 * pd - dd * (kd - 1) - 1
    numerator_h = H + 2 * ph - dh * (kh - 1) - 1
    numerator_w = W + 2 * pw - dw * (kw - 1) - 1

    if ceil_mode:
        D_out = (numerator_d + sd - 1) // sd + 1
        H_out = (numerator_h + sh - 1) // sh + 1
        W_out = (numerator_w + sw - 1) // sw + 1
    else:
        D_out = numerator_d // sd + 1
        H_out = numerator_h // sh + 1
        W_out = numerator_w // sw + 1

    total_outputs = N * C * D_out * H_out * W_out

    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_outputs, BLOCK_SIZE),)

    out = torch.empty(
        (N, C, D_out, H_out, W_out), dtype=input.dtype, device=input.device
    )
    indices = torch.empty(
        (N, C, D_out, H_out, W_out), dtype=torch.int32, device=input.device
    )

    max_pool3d_kernel[grid](
        input,
        out,
        indices,
        N,
        C,
        D,
        H,
        W,
        D_out,
        H_out,
        W_out,
        kernel_size=kd,
        stride=sd,
        padding=pd,
        dilation=dd,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out, indices)
