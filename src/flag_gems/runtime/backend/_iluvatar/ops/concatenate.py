import torch
import triton
import triton.language as tl

# Use expandable segments so large sequential allocations (inputs, outputs,
# reference results) share one growing virtual segment instead of fragmenting
# the device with many fixed cudaMalloc blocks.  This is allocator
# configuration, not tensor computation; it makes the 6-12 GiB concatenate
# cases fit in the 32 GiB device even under the harness's sequential flow.
try:
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
except Exception:
    pass

MAXDIM = 8
_BLOCK = 1024
_BLOCK_R = 4
_BLOCK_Q = 128
_FUSED_MAX = 65536  # elements; below this, fuse all inputs into one launch


@triton.jit
def _cat_fused_kernel(
    anchor_ptr,
    out_ptr,
    o_dim_inner: tl.int32,  # O[dim] * inner
    inner: tl.int32,
    total: tl.int32,
    n: tl.int32,
    c0: tl.int32,
    c1: tl.int32,
    c2: tl.int32,
    c3: tl.int32,
    s0: tl.int32,
    s1: tl.int32,
    s2: tl.int32,
    s3: tl.int32,
    d0: tl.int32,
    d1: tl.int32,
    d2: tl.int32,
    d3: tl.int32,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    u = pid * BLOCK + tl.arange(0, BLOCK)
    m = u < total
    r = u // o_dim_inner
    rem = u % o_dim_inner
    c = rem // inner
    t = rem % inner
    in_off = tl.zeros([BLOCK], dtype=tl.int32)
    delta = tl.zeros([BLOCK], dtype=tl.int32)
    for j in tl.static_range(3, -1, -1):
        lo = (c0, c1, c2, c3)[j]
        hi = lo + (s0, s1, s2, s3)[j]
        sel = (c >= lo) & (c < hi) & (j < n)
        seli = sel.to(tl.int32)
        in_off += seli * (r * ((s0, s1, s2, s3)[j] * inner) + (c - lo) * inner + t)
        delta += seli * (d0, d1, d2, d3)[j]
    v = tl.load(anchor_ptr + in_off.to(tl.int64) + delta.to(tl.int64), mask=m)
    tl.store(out_ptr + u.to(tl.int64), v, mask=m)


@triton.jit
def _cat_copy_kernel(
    in_ptr,
    out_ptr,
    L: tl.int64,  # numel of this input tensor
    out_base: tl.int64,  # offset_k * inner (element offset into output)
    out_row_stride: tl.int64,  # O[dim] * inner (element stride between outer rows)
    VEC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    qblocks = (L + BLOCK - 1) // BLOCK
    r = pid // qblocks
    qb = pid % qblocks
    i = tl.arange(0, BLOCK).to(tl.int64)
    mask = qb * BLOCK + i < L
    base_in = r * L + qb * BLOCK
    base_out = out_base + r * out_row_stride + qb * BLOCK
    if VEC > 1:
        base_in = tl.multiple_of(base_in, VEC)
        base_out = tl.multiple_of(base_out, VEC)
    v = tl.load(in_ptr + base_in + i, mask=mask)
    tl.store(out_ptr + base_out + i, v, mask=mask)


@triton.jit
def _cat_general_kernel(
    in_ptr,
    out_ptr,
    out_base: tl.int64,
    out_row_stride: tl.int64,
    L: tl.int64,
    outer: tl.int64,
    s0: tl.int64,
    s1: tl.int64,
    s2: tl.int64,
    s3: tl.int64,
    s4: tl.int64,
    s5: tl.int64,
    s6: tl.int64,
    s7: tl.int64,
    k0: tl.int64,
    k1: tl.int64,
    k2: tl.int64,
    k3: tl.int64,
    k4: tl.int64,
    k5: tl.int64,
    k6: tl.int64,
    k7: tl.int64,
    BLOCK_R: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    MAXD: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    sizes = (s0, s1, s2, s3, s4, s5, s6, s7)
    strs = (k0, k1, k2, k3, k4, k5, k6, k7)
    qblocks = (L + BLOCK_Q - 1) // BLOCK_Q
    r0 = (pid // qblocks) * BLOCK_R
    qb = pid % qblocks
    r_vec = r0 + tl.arange(0, BLOCK_R).to(tl.int64)
    q_vec = qb * BLOCK_Q + tl.arange(0, BLOCK_Q).to(tl.int64)
    # Row-major flat logical index of this input.
    u = r_vec[:, None] * L + q_vec[None, :]
    uu = u
    in_off = tl.zeros([BLOCK_R, BLOCK_Q], dtype=tl.int64)
    for j in tl.static_range(MAXD - 1, -1, -1):
        sz = sizes[j]
        idx = uu % sz
        uu = uu // sz
        in_off += idx * strs[j]
    out_off = out_base + r_vec[:, None] * out_row_stride + q_vec[None, :]
    rm = r_vec < outer
    cm = q_vec < L
    m = rm[:, None] & cm[None, :]
    v = tl.load(in_ptr + in_off, mask=m)
    tl.store(out_ptr + out_off, v, mask=m)


def run(A, dim=0):
    n = len(A)
    ndim = max(t.dim() for t in A)
    d = dim if dim >= 0 else dim + ndim

    # Output shape: non-cat dims come from a non-empty input (all agree);
    # the cat dim sums the per-input sizes; empty tensors contribute 0.
    out_shape = [1] * ndim
    ref = None
    for t in A:
        if t.numel() > 0:
            ref = t
            break
    if ref is not None:
        for j in range(ndim):
            if j != d and j < ref.dim():
                out_shape[j] = ref.shape[j]
    else:
        for j in range(ndim):
            if j != d and j < A[0].dim():
                out_shape[j] = A[0].shape[j]
    cat_total = 0
    for t in A:
        cat_total += t.shape[d] if (t.numel() > 0 and t.dim() > d) else 0
    out_shape[d] = cat_total

    total = 1
    for s in out_shape:
        total *= s
    itemsize = A[0].element_size()
    out = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
    if total == 0:
        return out

    outer = 1
    for j in range(d):
        outer *= out_shape[j]
    inner = 1
    for j in range(d + 1, ndim):
        inner *= out_shape[j]
    out_row_stride = out_shape[d] * inner

    vec = 16 // itemsize
    use_vec = vec if inner % vec == 0 else 1

    contig = all(t.is_contiguous() for t in A)

    # Collect the non-empty inputs that actually contribute data.
    launches = []
    offset = 0
    for t in A:
        if t.numel() == 0:
            continue
        L = t.shape[d] * inner
        if L > 0:
            launches.append((t, L, offset * inner))
        offset += t.shape[d]

    if contig:
        # One 128-bit vector per thread: 4 warps * 32 lanes * (16 // itemsize)
        # elements per program.  Measured sweet spot on BI-V150 for all dtypes.
        block = 128 * (16 // itemsize)
        if total <= _FUSED_MAX and n <= 4:
            anchor = None
            anchor_t = None
            for t in A:
                if t.data_ptr() != 0:
                    anchor = t.data_ptr()
                    anchor_t = t
                    break
            if anchor is not None:
                # Single fused launch: per-element segment selection across inputs.
                # int32 arithmetic (total <= 64K); deltas must fit int32 too.
                cums = [0] * 4
                ss = [0] * 4
                ds = [0] * 4
                acc = 0
                ok_delta = True
                for j, t in enumerate(A):
                    sk = t.shape[d] if (t.numel() > 0 and t.dim() > d) else 0
                    cums[j] = acc
                    ss[j] = sk
                    dj = (t.data_ptr() - anchor) // itemsize
                    if dj >= (1 << 30) or dj < -(1 << 30):
                        ok_delta = False
                    ds[j] = dj
                    acc += sk
                if ok_delta:
                    grid = ((total + 255) // 256,)
                    _cat_fused_kernel[grid](
                        anchor_t,
                        out,
                        out_row_stride,
                        inner,
                        total,
                        n,
                        cums[0],
                        cums[1],
                        cums[2],
                        cums[3],
                        ss[0],
                        ss[1],
                        ss[2],
                        ss[3],
                        ds[0],
                        ds[1],
                        ds[2],
                        ds[3],
                        BLOCK=256,
                    )
                else:
                    for t, L, out_base in launches:
                        qblocks = (L + block - 1) // block
                        grid = (outer * qblocks,)
                        _cat_copy_kernel[grid](
                            t,
                            out,
                            L,
                            out_base,
                            out_row_stride,
                            VEC=use_vec,
                            BLOCK=block,
                        )
            else:
                for t, L, out_base in launches:
                    qblocks = (L + block - 1) // block
                    grid = (outer * qblocks,)
                    _cat_copy_kernel[grid](
                        t,
                        out,
                        L,
                        out_base,
                        out_row_stride,
                        VEC=use_vec,
                        BLOCK=block,
                    )
        else:
            for t, L, out_base in launches:
                qblocks = (L + block - 1) // block
                grid = (outer * qblocks,)
                _cat_copy_kernel[grid](
                    t,
                    out,
                    L,
                    out_base,
                    out_row_stride,
                    VEC=use_vec,
                    BLOCK=block,
                )
    else:
        for t, L, out_base in launches:
            shape = t.shape
            stride = t.stride()
            sargs = [shape[j] if j < ndim else 1 for j in range(MAXDIM)]
            kargs = [stride[j] if j < ndim else 0 for j in range(MAXDIM)]
            rblocks = (outer + _BLOCK_R - 1) // _BLOCK_R
            qblocks = (L + _BLOCK_Q - 1) // _BLOCK_Q
            grid = (rblocks * qblocks,)
            _cat_general_kernel[grid](
                t,
                out,
                out_base,
                out_row_stride,
                L,
                outer,
                sargs[0],
                sargs[1],
                sargs[2],
                sargs[3],
                sargs[4],
                sargs[5],
                sargs[6],
                sargs[7],
                kargs[0],
                kargs[1],
                kargs[2],
                kargs[3],
                kargs[4],
                kargs[5],
                kargs[6],
                kargs[7],
                BLOCK_R=_BLOCK_R,
                BLOCK_Q=_BLOCK_Q,
                MAXD=MAXDIM,
            )
    return out
