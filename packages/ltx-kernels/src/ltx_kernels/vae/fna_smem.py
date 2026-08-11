"""SMEM operand fills and views shared by the DiffVAE CuTe DSL kernels.
Extracted from :mod:`ltx_kernels.vae.fna_attn_core` so the attention half and the
TMA/UMMA helpers can stay focused. Constants: :mod:`ltx_kernels.vae.fna_types`.
"""

# NOTE: no `from __future__ import annotations` -- cute.struct needs real types.

import cutlass
import cutlass.cute as cute

from ltx_kernels.vae.fna_types import ACC, HD, IO, VEC, VEC_ROWS, M

# --------------------------------------------------------------------------
# SMEM operand fills. A/B UMMA layouts are ``((MN, 16), 1, K_TILE/16, stages)``
# with the K coordinate split as ``k = st*64 + kb*16 + ki``. These are plain
# Python (no device branch), so they trace inline with ordinary ``range``.
# --------------------------------------------------------------------------

# A whole ``ki`` run is one contiguous 32B group of the swizzled operand, so slicing
# it out and handing that to ``autovec_copy`` gives 16B stores while keeping the
# swizzle attached. Element-wise access instead costs 64 ``STS.U16`` per stage.
# Slicing a *sub*-run does not work: it lands mid-granule and corrupts the operand.
_KRUN = 16


def _fill_a_chunk(sA, row, vals, c0, n):
    """Write ``vals`` into channels ``[c0, c0+n)`` of A-operand row ``row``.
    ``c0`` and ``n`` must be compile-time and multiples of ``_KRUN``: a dynamic
    coordinate loses the slice's static alignment and ``autovec_copy`` falls back to
    scalar stores. That is why callers spell the warpgroup split as a constexpr loop
    over ``g`` guarded by ``wg == g`` rather than as an arithmetic offset.
    """
    frag = cute.make_rmem_tensor(cute.make_layout(_KRUN), IO)
    for k in range(n // _KRUN):
        c = c0 + k * _KRUN
        for ki in range(_KRUN):
            frag[ki] = IO(vals[k * _KRUN + ki])
        cute.autovec_copy(frag, sA[((row, None), 0, (c % 64) // 16, c // 64)])


def _read_a_chunk(sA, row, vals, c0, n):
    """Read channels ``[c0, c0+n)`` of A-operand row ``row`` back into registers."""
    frag = cute.make_rmem_tensor(cute.make_layout(_KRUN), IO)
    for k in range(n // _KRUN):
        c = c0 + k * _KRUN
        cute.autovec_copy(sA[((row, None), 0, (c % 64) // 16, c // 64)], frag)
        for ki in range(_KRUN):
            vals[k * _KRUN + ki] = ACC(frag[ki])


def _vec_ptr(t, coord):
    """A ``VEC``-wide, 16B-aligned pointer at ``(row, col)`` of a row-major 2D tensor.
    Adding a dynamic offset to a pointer drops its alignment to one element, and
    ``autovec_copy`` then picks its vector width from that -- 16 scalar ``LDG.E.U16``
    instead of one ``LDG.E.128``. Every offset reached here is a multiple of ``VEC``,
    so re-asserting the alignment is sound.
    The linear offset is built in 64-bit rather than via ``crd2idx``, whose int32
    arithmetic wraps once the tensor passes 2**31 elements: a DiffVAE stage-4
    activation at 4K is ``(4879680, 512)`` = 2.5e9 elements, and the wrapped
    (negative) offset reads or writes unmapped memory -- decoding 3840x2176 faulted
    with CUDA_ERROR_ILLEGAL_ADDRESS here.
    """
    row, col = coord
    row_stride, col_stride = t.layout.stride
    off = cutlass.Int64(row) * cutlass.Int64(row_stride) + cutlass.Int64(col) * cutlass.Int64(col_stride)
    return (t.iterator + off).align(16)


def _row_vec_ptr(t, row, col, n: cutlass.Constexpr):
    """16B-aligned pointer at ``(row, col)`` of a row-major ``(rows, n)`` tensor.
    64-bit offset for the same reason as :func:`_vec_ptr`: K/V here are
    ``(tokens*NH, HD)``, which is 2.5e9 elements at a 4K stage-4 tile.
    """
    return (t.iterator + (cutlass.Int64(row) * cutlass.Int64(n) + cutlass.Int64(col))).align(16)


def _fill_a_gather(sA, src, rowix, vlane, c_len, c_off=0):
    """Fill all ``c_len`` channels of the A operand from a global ``(row, col)`` tensor.
    ``rowix[t]`` is the global row feeding A-row ``t``. Each thread takes ``VEC``
    consecutive columns of ``VEC_ROWS`` rows at a time, so a load is 16B per thread
    and contiguous across the threads covering a row -- and the same run is
    contiguous in the swizzled SMEM layout. The ``M`` threads span ``M`` columns, so
    a wider activation needs one pass per ``M``-channel group.
    ``c_off`` shifts the *source* window without moving the destination, which lets
    the NA kernel fill a four-head operand from head group ``hg`` of a much wider Q
    tensor. It may be a runtime value (every offset reaching it is a multiple of
    ``VEC``) and folds away at the default 0.
    """
    c0, r0 = vlane
    frag = cute.make_rmem_tensor(cute.make_layout(VEC), IO)
    for kc in range(c_len // M):
        for nb in range(M // VEC_ROWS):
            t = nb * VEC_ROWS + r0
            row = rowix[t]
            c = kc * M + c0
            cute.autovec_copy(
                cute.make_tensor(_vec_ptr(src, (row, c + c_off)), cute.make_layout(VEC)),
                frag,
            )
            cute.autovec_copy(frag, sA[((t, None), 0, (c % 64) // 16, c // 64)])


def _scatter_a(sA, dst, rowix, vlane, c_len, c_off=0):
    """Write the A operand back out to a global ``(row, col)`` tensor, coalesced.
    Mirror of :func:`_fill_a_gather`, ``c_off`` included. The per-thread-row
    alternative touches a different cache line per lane.
    """
    c0, r0 = vlane
    frag = cute.make_rmem_tensor(cute.make_layout(VEC), IO)
    for kc in range(c_len // M):
        for nb in range(M // VEC_ROWS):
            t = nb * VEC_ROWS + r0
            row = rowix[t]
            c = kc * M + c0
            cute.autovec_copy(sA[((t, None), 0, (c % 64) // 16, c // 64)], frag)
            cute.autovec_copy(
                frag,
                cute.make_tensor(_vec_ptr(dst, (row, c + c_off)), cute.make_layout(VEC)),
            )


def _load_head_row(src, row, frag):
    """One head's K or V row into registers, ``VEC`` bf16 per load.
    Split from the SMEM store so the head loop can be software-pipelined: head
    ``nh+1``'s row is fetched right after head ``nh``'s is published, which puts its
    latency behind head ``nh``'s MMAs and softmax instead of in front of them.
    """
    fz = cute.zipped_divide(frag, cute.make_layout(VEC))
    for k in range(HD // VEC):
        cute.autovec_copy(
            cute.make_tensor(_row_vec_ptr(src, row, k * VEC, HD), cute.make_layout(VEC)),
            fz[None, k],
        )


def _store_k_row(frag, sK, n, st_off):
    """Registers -> K-major B operand row ``n``. One ``_KRUN`` per swizzle granule."""
    fz = cute.zipped_divide(frag, cute.make_layout(_KRUN))
    for k in range(HD // _KRUN):
        cute.autovec_copy(fz[None, k], sK[((n, None), 0, k, st_off)])


def _store_v_row(frag, sV, key, st_off):
    """Registers -> MN-major B operand key ``key``.
    MN-major puts the 64 channels contiguous, so the whole row is a single slice of
    the operand rather than the 64 ``STS.U16`` a per-channel loop emits.
    """
    cute.autovec_copy(frag, sV[((None, key % 64 % 16), 0, key % 64 // 16, st_off + key // 64)])


def _store_head_row(dst, row, vals, n=HD, c0: int = 0):
    """Write ``n`` accumulator values to ``dst[row, c0:c0+n]``, vectorised.
    ``dst`` rows are always ``HD`` wide; ``c0``/``n`` let a warpgroup that owns only
    part of a head write just its slice.
    """
    frag = cute.make_rmem_tensor(cute.make_layout(VEC), IO)
    for k in range(n // VEC):
        for u in range(VEC):
            frag[u] = IO(vals[k * VEC + u])
        cute.autovec_copy(
            frag,
            cute.make_tensor(_row_vec_ptr(dst, row, c0 + k * VEC, HD), cute.make_layout(VEC)),
        )


def _alias(buf, layout, off: int = 0):
    """A UMMA-operand view over ``buf``'s storage, ``off`` elements in.
    tcgen05 wants the swizzle on the pointer rather than folded into the layout, so
    the composed layout is split apart here the way ``SmemAllocator.allocate_tensor``
    would have done it. Re-asserting the alignment matters: adding an offset drops a
    pointer's static alignment and a swizzled operand needs its 128 bytes back.
    """
    ptr = buf.iterator if off == 0 else (buf.iterator + off).align(128)
    return cute.make_tensor(cute.recast_ptr(ptr, layout.inner, dtype=IO), layout.outer)


def cooperative_copy_slot(tidx):
    """This thread's ``(first_column, row_in_step)`` slot in the operand copies.
    :func:`_fill_a_gather` and :func:`_scatter_a` move ``VEC`` consecutive columns of
    ``VEC_ROWS`` rows per step; this is the column range and the within-step row that
    thread ``tidx`` handles in either.
    """
    return VEC * (tidx % (M // VEC)), tidx // (M // VEC)


def _publish_smem():
    """Make this CTA's SMEM stores visible to the async (tcgen05/TMA) proxy.
    ``bar.sync`` orders thread against thread; it says nothing about the async proxy
    the UMMA reads its operands through, so an operand written by ordinary stores
    needs ``fence.proxy.async.shared::cta`` before the barrier that publishes it.
    Without it the kernel is silently *non-deterministic* -- output stops being
    bit-identical run to run and PSNR drifts a couple of dB, which no correctness
    tolerance catches. The ``test_is_deterministic`` case in each kernel's test
    module does.
    """
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
