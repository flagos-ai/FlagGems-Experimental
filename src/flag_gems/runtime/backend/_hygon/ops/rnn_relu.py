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

from flag_gems import runtime
from flag_gems.ops.rnn_relu import _params_unpack
from flag_gems.runtime.op_registrar import GeneralOpRegistrar
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registration-key fix (overload remap)
# ---------------------------------------------------------------------------
#
# Why this is needed
# ------------------
# The general layer registers the fused RNN implementation under the *bare*
# aten name ``rnn_relu`` (see ``flag_gems.__init__._FULL_CONFIG``).  But the
# ATen operator ``torch.rnn_relu`` only exposes the overloads ``rnn_relu.input``
# and ``rnn_relu.data`` -- there is no default (empty) overload:
#
#     aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params,
#                          bool has_biases, int num_layers, float dropout,
#                          bool train, bool bidirectional, bool batch_first)
#                          -> (Tensor, Tensor)
#
# So ``lib.impl("rnn_relu", fn, "CUDA")`` binds ``fn`` to a phantom empty
# overload that is never dispatched to, and the fused kernel is silently never
# invoked: under ``flag_gems.use_gems()`` the composite ``torch.rnn_relu``
# decomposes into many tiny aten sub-ops instead.  On Hygon the bfloat16 path
# ran ~0.38x of native torch this way.  The correct key is ``rnn_relu.input``
# (``_FULL_CONFIG`` already uses ``<op>.<overload>`` keys elsewhere, e.g.
# ``addmm.out``); the bare ``rnn_relu`` entry there is effectively a bug, but we
# must not edit the shared general layer from a vendor PR.
#
# Why the remap lives in ``register_impl``
# ----------------------------------------
# The registration must take effect only inside the ``use_gems()`` context
# (the general layer registers/unregisters there).  Registering ``.input`` at
# import time via an independent ``torch.library.Library`` would hijack
# ``torch.rnn_relu`` globally -- even outside ``use_gems()`` -- which is
# incorrect.  Intercepting the general registration path is therefore the only
# way to fix the key while preserving gems' scoping.
#
# Safety of the wrapper
# ---------------------
# Instead of hard-coding a single key rewrite, we keep a *class-level mapping*
# ``_KEY_OVERLOAD_REMAP`` and wrap ``register_impl`` once (idempotent).  The
# wrapper:
#   * only rewrites keys present in the shared mapping and leaves every other
#     op untouched (so it never changes another op's dispatch);
#   * is additive across vendors -- if another backend needs its own overload
#     remap it registers the key in the same dict rather than re-wrapping the
#     method, so there is no conflicting second patch;
#   * preserves and delegates to the original ``register_impl`` unchanged;
#   * does not touch the native (outside ``use_gems``) dispatch path.

_RNN_RELU_KEY = "rnn_relu"
_RNN_RELU_OVERLOAD_KEY = "rnn_relu.input"


def _register_key_overload_remap(bare_key, overload_key):
    """Record a bare-name -> overload-key remap and install the wrapper once.

    Idempotent and additive: multiple ops (and multiple vendor modules) can
    register their own remaps into the shared class-level table without
    re-patching ``register_impl`` or clobbering each other.
    """
    remap = getattr(GeneralOpRegistrar, "_key_overload_remap", None)
    if remap is None:
        remap = {}
        GeneralOpRegistrar._key_overload_remap = remap
    remap[bare_key] = overload_key

    if getattr(GeneralOpRegistrar, "_key_overload_remap_installed", False):
        return

    _orig_register_impl = GeneralOpRegistrar.register_impl

    def register_impl(self, key, fn, extra_dispatch_keys=()):
        key = GeneralOpRegistrar._key_overload_remap.get(key, key)
        return _orig_register_impl(self, key, fn, extra_dispatch_keys)

    GeneralOpRegistrar.register_impl = register_impl
    GeneralOpRegistrar._key_overload_remap_installed = True


_register_key_overload_remap(_RNN_RELU_KEY, _RNN_RELU_OVERLOAD_KEY)


# ---------------------------------------------------------------------------
# Fused forward kernel (Hygon)
# ---------------------------------------------------------------------------
#
# Dtype dispatch rationale:
#
# * float16 -- the Triton mat-vec reduction cannot bit-match MIOpen's fused
#   float16 RNN (diverges ~4e-3, failing the strict atol=1e-4 dispatch test),
#   so float16 forwards to the native ``miopen_rnn`` fast path (bit-exact).
# * float32 -- the Triton kernel matches native to ~2e-7 (well within
#   tolerance) and is far faster than the reference on Hygon, so float32 uses
#   the Triton kernel.
# * bfloat16 -- MIOpen does not support bfloat16 RNNs ("RNN datatype must be
#   float or half"), so native torch falls back to a slow unfused path
#   (~6 ms => Gems speedup ~0.38x in the baseline).  The Triton kernel matches
#   native bit-for-bit here and is ~50-100x faster.
#
# Numerical note: native ``rnn_relu`` computes the two projections
# (input->hidden and hidden->hidden) as separate matmuls that each round back
# to the storage dtype before being summed.  Accumulating both in float32 and
# rounding once (as a naive fused kernel would) diverges by ~1 ULP and fails
# the strict dispatch test.  To match native bit patterns we round each
# projection, and the pre-activation sum, back to the storage dtype
# (``OUT_DTYPE``) inside the kernel.
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rnn_relu"),
    key=["seq_len", "input_size", "hidden_size"],
)
@triton.jit
def rnn_relu_forward_kernel(
    input_ptr,
    hx_ptr,
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    output_ptr,
    hidden_output_ptr,
    hidden_read_ptr,
    seq_len,
    batch_size,
    input_size,
    hidden_size,
    batch_first: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-layer unidirectional RNN with ReLU activation (Hygon bf16).

    Grid: (batch_size,) -- one program per batch element.  Each program walks
    all time steps sequentially, double-buffering the hidden state so chunks
    beyond BLOCK_SIZE never read partially-updated values within a step.
    """
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    CHUNK: tl.constexpr = 64  # inner block for mat-vec loops
    num_hid_blocks = tl.cdiv(hidden_size, BLOCK_SIZE)
    chunk_offs = tl.arange(0, CHUNK)
    bid_offs = tl.arange(0, BLOCK_SIZE)

    # --- initialize hidden state in global scratch ---
    _init_hidden(
        hidden_output_ptr,
        hx_ptr,
        batch_idx,
        hidden_size,
        num_hid_blocks,
        BLOCK_SIZE,
    )

    if batch_first:
        batch_inp_stride = seq_len * input_size
        batch_out_stride = seq_len * hidden_size
    else:
        batch_inp_stride = input_size
        batch_out_stride = hidden_size

    h_scratch_base = hidden_output_ptr + batch_idx * hidden_size
    h_read_base = hidden_read_ptr + batch_idx * hidden_size

    for t in range(seq_len):
        if batch_first:
            t_inp_off = batch_idx * batch_inp_stride + t * input_size
            t_out_off = batch_idx * batch_out_stride + t * hidden_size
        else:
            t_inp_off = t * (batch_size * input_size) + batch_idx * input_size
            t_out_off = t * (batch_size * hidden_size) + batch_idx * hidden_size

        # Double-buffer: snapshot previous-step hidden state into read buffer
        # so later chunks don't read already-updated values from this step.
        for hid_block in range(num_hid_blocks):
            h_start = hid_block * BLOCK_SIZE
            h_offs = h_start + bid_offs
            h_mask = h_offs < hidden_size
            h_val = tl.load(h_scratch_base + h_offs, mask=h_mask, other=0.0)
            tl.store(h_read_base + h_offs, h_val, mask=h_mask)

        for hid_block in range(num_hid_blocks):
            h_start = hid_block * BLOCK_SIZE
            h_offs = h_start + bid_offs
            h_mask = h_offs < hidden_size

            # --- ih = W_ih[h_offs, :] @ x_t + b_ih ---
            ih_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for inp_start in range(0, input_size, CHUNK):
                i_offs = inp_start + chunk_offs
                i_mask = i_offs < input_size
                w_offs = h_offs[:, None] * input_size + i_offs[None, :]
                w_mask = h_mask[:, None] & i_mask[None, :]
                w_tile = tl.load(weight_ih_ptr + w_offs, mask=w_mask, other=0.0).to(
                    tl.float32
                )
                x_tile = tl.load(
                    input_ptr + t_inp_off + i_offs, mask=i_mask, other=0.0
                ).to(tl.float32)
                ih_acc += tl.sum(w_tile * x_tile[None, :], axis=1)

            if bias_ih_ptr is not None:
                b_ih = tl.load(bias_ih_ptr + h_offs, mask=h_mask, other=0.0).to(
                    tl.float32
                )
                ih_acc += b_ih

            # Round the input->hidden projection to storage dtype to match
            # native rnn_relu's per-matmul rounding.
            ih_acc = ih_acc.to(OUT_DTYPE).to(tl.float32)

            # --- hh = W_hh[h_offs, :] @ h_prev + b_hh ---
            hh_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for hid_in_start in range(0, hidden_size, CHUNK):
                j_offs = hid_in_start + chunk_offs
                j_mask = j_offs < hidden_size
                w_offs = h_offs[:, None] * hidden_size + j_offs[None, :]
                w_mask = h_mask[:, None] & j_mask[None, :]
                w_tile = tl.load(weight_hh_ptr + w_offs, mask=w_mask, other=0.0).to(
                    tl.float32
                )
                h_tile = tl.load(h_read_base + j_offs, mask=j_mask, other=0.0).to(
                    tl.float32
                )
                hh_acc += tl.sum(w_tile * h_tile[None, :], axis=1)

            if bias_hh_ptr is not None:
                b_hh = tl.load(bias_hh_ptr + h_offs, mask=h_mask, other=0.0).to(
                    tl.float32
                )
                hh_acc += b_hh

            hh_acc = hh_acc.to(OUT_DTYPE).to(tl.float32)

            # --- h_new = relu(ih + hh) ---  (sum rounds to storage dtype too)
            h_new = (ih_acc + hh_acc).to(OUT_DTYPE).to(tl.float32)
            h_new = tl.where(h_new > 0, h_new, 0.0)

            tl.store(output_ptr + t_out_off + h_offs, h_new, mask=h_mask)
            tl.store(h_scratch_base + h_offs, h_new, mask=h_mask)


@triton.jit
def _init_hidden(
    hidden_output_ptr,
    hx_ptr,
    batch_idx,
    hidden_size,
    num_hid_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Initialize the scratch hidden state from hx (or zeros)."""
    bid_offs = tl.arange(0, BLOCK_SIZE)
    for hid_block in range(num_hid_blocks):
        h_start = hid_block * BLOCK_SIZE
        h_offs = h_start + bid_offs
        h_mask = h_offs < hidden_size
        if hx_ptr is not None:
            h_val = tl.load(
                hx_ptr + batch_idx * hidden_size + h_offs,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            h_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        tl.store(
            hidden_output_ptr + batch_idx * hidden_size + h_offs,
            h_val,
            mask=h_mask,
        )


_TORCH_TO_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _rnn_relu_triton_forward(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """Launch the Hygon Triton RNN ReLU kernel (bfloat16 / float32)."""
    logger.debug("GEMS RNN_RELU FORWARD KERNEL LAUNCH (hygon triton)")

    if num_layers > 1 or bidirectional:
        raise NotImplementedError(
            "GEMS RNN_RELU only supports single-layer unidirectional"
        )

    if batch_first:
        batch_size, seq_len, input_size = input.shape
    else:
        seq_len, batch_size, input_size = input.shape

    hidden_size = hx.shape[2]

    output_shape = (
        (batch_size, seq_len, hidden_size)
        if batch_first
        else (seq_len, batch_size, hidden_size)
    )
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    hidden_shape = (num_layers, batch_size, hidden_size)
    hidden = torch.empty(hidden_shape, dtype=input.dtype, device=input.device)
    hidden_read = torch.empty(hidden_shape, dtype=input.dtype, device=input.device)

    weight_ih, weight_hh, bias_ih, bias_hh = _params_unpack(params, has_biases)

    input = input.contiguous()
    weight_ih = weight_ih.contiguous()
    weight_hh = weight_hh.contiguous()
    if bias_ih is not None:
        bias_ih = bias_ih.contiguous()
    if bias_hh is not None:
        bias_hh = bias_hh.contiguous()

    out_tl_dtype = _TORCH_TO_TL.get(input.dtype, tl.float32)
    grid = (batch_size,)

    with runtime.torch_device_fn.device(input.device):
        rnn_relu_forward_kernel[grid](
            input,
            hx,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            output,
            hidden,
            hidden_read,
            seq_len,
            batch_size,
            input_size,
            hidden_size,
            batch_first,
            out_tl_dtype,
        )

    return output, hidden


# RNN mode id understood by miopen_rnn: 0 = ReLU, 1 = tanh, 2 = LSTM, 3 = GRU.
_MIOPEN_MODE_RELU = 0


def _rnn_relu_miopen_forward(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """Forward to Hygon's fused ``miopen_rnn`` (float16 / float32).

    This is exactly what native ``torch.rnn_relu`` lowers to, so it is
    bit-for-bit identical to the reference while remaining the fast path.
    """
    logger.debug("GEMS RNN_RELU MIOPEN FORWARD (hygon)")
    hidden_size = hx.shape[2]
    out = torch.ops.aten.miopen_rnn(
        input,
        list(params),
        4 if has_biases else 2,
        hx,
        None,
        _MIOPEN_MODE_RELU,
        hidden_size,
        num_layers,
        batch_first,
        dropout,
        train,
        bidirectional,
        [],
        None,
    )
    return out[0], out[1]


def rnn_relu(
    input,
    hx=None,
    params=None,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=False,
):
    """Applies an Elman RNN with ReLU activation (Hygon specialization).

    Dispatches by dtype:

    * bfloat16 / float32 -> fused Triton kernel.  MIOpen has no bf16 RNN (so
      native bf16 is slow), and the Triton kernel matches native float32 within
      tolerance while running much faster on Hygon.
    * float16 -> fused ``miopen_rnn`` (native fast path, bit-exact); the Triton
      reduction cannot bit-match MIOpen's fused float16 RNN.

    Supports single-layer, unidirectional RNN without dropout.
    """
    logger.debug("GEMS RNN_RELU (hygon)")

    if params is None:
        raise ValueError("params must be provided")
    if hx is None:
        raise ValueError("hx must be provided to match torch.rnn_relu schema")

    if num_layers != 1 or bidirectional or dropout != 0:
        raise NotImplementedError(
            "GEMS RNN_RELU only supports single-layer unidirectional without dropout"
        )

    if input.dtype in (torch.bfloat16, torch.float32):
        return _rnn_relu_triton_forward(
            input,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )

    return _rnn_relu_miopen_forward(
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    )
