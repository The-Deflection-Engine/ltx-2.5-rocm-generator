"""Primitives shared by the DiffVAE CuTe DSL kernels, plus the attention half itself.
``block_fna_dsl`` (the fused ``DiffusionNABlock``) and ``na_attn_dsl`` (standalone
neighborhood attention) are separate kernels -- their prologues and epilogues have
nothing in common -- but they share every operand fill, the TMEM plumbing, the MMA
wrappers, the RoPE tables and, in :func:`attention_tile`, the windowed-NA loop. One
copy of the mask, the software pipeline and the softmax, so the two cannot drift.
This module holds host helpers, RoPE, CTA barriers, and the attention half
(:func:`attention_tile`). Constants: :mod:`ltx_kernels.vae.fna_types`. SMEM operand
fills: :mod:`ltx_kernels.vae.fna_smem`. TMEM / TMA-UMMA GEMM:
:mod:`ltx_kernels.vae.fna_gemm`. The coordinate arithmetic both kernels share is
:mod:`ltx_kernels.vae.fna_geometry`, which this imports and which imports nothing from
here -- worth reading first, since it defines the box vocabulary (volume / slab core /
slab box / query tile / halo panel / NA window) the comments below use.
Datacenter Blackwell only: everything here is built on ``tcgen05`` UMMA and Tensor
Memory. Several DSL constraints shape the code in non-obvious ways; each is noted at
the helper it constrains.
"""

# NOTE: no `from __future__ import annotations` -- cute.struct needs real types.

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack

from ltx_kernels.vae.fna_gemm import _mma_issue, _tmem_load, _tmem_store
from ltx_kernels.vae.fna_geometry import at_most, flatten_3d, unflatten_3d
from ltx_kernels.vae.fna_smem import (
    _fill_a_chunk,
    _load_head_row,
    _publish_smem,
    _store_k_row,
    _store_v_row,
)
from ltx_kernels.vae.fna_types import (
    ACC,
    ACC_COLS,
    HD,
    HEAD_COLS,
    LOG2E,
    NUM_WARPGROUPS,
    THREADS,
    WARPGROUPS_PER_HEAD,
    M,
)


def default_tile_thw(kernel_size: "tuple[int, int, int]") -> "tuple[int, int, int]":
    """Pick the ``M``-token query box minimising halo-panel tokens per query."""
    kt, kh, kw = kernel_size
    best, best_panel = (2, 8, 8), 1 << 60
    for tt in (1, 2, 4, 8, 16):
        for th in (2, 4, 8, 16, 32):
            if M % (tt * th) != 0:
                continue
            tw = M // (tt * th)
            if tw < 1 or tw > 32:
                continue
            panel = (tt + kt - 1) * (th + kh - 1) * (tw + kw - 1)
            if panel < best_panel:
                best, best_panel = (tt, th, tw), panel
    return best


def cta_count(n_tiles: int, device: torch.device) -> int:
    """Persistent grid size: one CTA per SM, capped by the work available.
    ``block_fna_dsl``'s device-wide barrier requires every CTA to be resident, so the
    grid may never exceed the SM count.
    """
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    return int(max(1, min(n_tiles, sm)))


def _dyn(t: torch.Tensor):
    return from_dlpack(t).mark_layout_dynamic()


def _dyn_act(t: torch.Tensor):
    """An activation tensor whose rows the cooperative copies read 16B at a time."""
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic()


def _dyn_w(t: torch.Tensor):
    """Mark a weight matrix for TMA: contiguous K axis with known divisibility."""
    return (
        from_dlpack(t, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=int(t.shape[1]))
    )


def _as_rows(t: torch.Tensor) -> torch.Tensor:
    """``(T, H, W, C)`` -> ``(T*H*W, C)``, without copying where possible.
    Only the channel axis has to be contiguous -- the kernel reads rows through the
    layout's dynamic stride -- plus the 16B alignment the cooperative copies assume.
    Forcing full contiguity would copy the whole volume.
    """
    if t.stride(-1) != 1:
        t = t.contiguous()
    t = t.flatten(0, 2)
    if t.data_ptr() % 16 != 0 or (t.stride(0) * t.element_size()) % 16 != 0:
        t = t.contiguous()
    return t


@cute.jit
def panel_key_row(
    kv_tile,
    lane,
    panel,
    n_panel_keys: cutlass.Constexpr,
    panel_hw: cutlass.Constexpr,
    panel_w: cutlass.Constexpr,
):
    """K/V-source row of the key lane ``lane`` owns in KV tile ``kv_tile``.
    A panel's keys are walked row-major, ``M`` per KV tile, so this lane's key sits at
    flat panel offset ``kv_tile * M + lane``. Turning that into a source row takes two
    steps: unflatten it to a panel-local ``(dt, dh, dw)``, then re-flatten the
    *absolute* position against the extents of the box the source was filled over.
    ``panel`` is ``(panel_t0, panel_h0, panel_w0, box_t0, box_h0, box_w0, box_h,
    box_w)``: the halo panel's origin in the volume, plus the origin and the H/W
    extents of that source box. The fused kernel's box is the slab box, so those are
    its own; the standalone NA kernel reads the caller's whole-volume K/V, so its box
    origin is zero and its extents are H and W. Same arithmetic either way.
    Hoisted out of the KV loop because the head pipeline needs the *next* tile's row
    one head early. A lane past the end of the panel reads a clamped row; the mask
    discards its score.
    """
    panel_t0, panel_h0, panel_w0, box_t0, box_h0, box_w0, box_h, box_w = panel
    key = at_most(kv_tile * M + lane, n_panel_keys - 1)
    d_t, d_h, d_w = unflatten_3d(key, panel_hw, panel_w)
    return flatten_3d(
        panel_t0 - box_t0 + d_t,
        panel_h0 - box_h0 + d_h,
        panel_w0 - box_w0 + d_w,
        box_h,
        box_w,
    )


@cute.jit
def grid_barrier(bar, tidx, n_ctas, phase):
    """Sense-reversing device-wide barrier; every CTA must be resident.
    ``bar[0]`` counts arrivals, ``bar[1]`` is the sense. The grid is capped at the SM
    count (see :func:`cta_count`) so no CTA can wait for one that was never
    scheduled.
    """
    cute.arch.fence_acq_rel_gpu()
    cute.arch.sync_threads()
    if tidx == 0:
        old = cute.arch.atomic_add(bar.iterator, cutlass.Int32(1), sem="acq_rel", scope="gpu")
        if old == n_ctas - 1:
            cute.arch.atomic_exch(bar.iterator, cutlass.Int32(0), sem="release", scope="gpu")
            cute.arch.atomic_exch(bar.iterator + 1, phase + 1, sem="release", scope="gpu")
        else:
            while cute.arch.atomic_add(bar.iterator + 1, cutlass.Int32(0), sem="acquire", scope="gpu") == phase:
                pass
    cute.arch.sync_threads()
    cute.arch.fence_acq_rel_gpu()


# Row reductions across the warpgroups sharing a row. Each holds a different slice of
# the row's columns, so the RMSNorm sums have to be combined. The barrier inside means
# these must be called by the *whole* CTA -- never from inside a ``wg == g`` arm. The
# softmax needs no such reduction; see :func:`attention_tile`.


@cute.jit
def row_sum(buf, lane, wg, v):
    """Sum ``v`` across all ``NUM_WARPGROUPS`` warpgroups holding row ``lane``."""
    buf[lane, wg] = v
    cute.arch.sync_threads()
    r = buf[lane, 0]
    for g in range(1, NUM_WARPGROUPS):
        r = r + buf[lane, g]
    return r


@cute.jit
def head_sum2(buf, lane, wg, v0, v1):
    """Two row sums, reduced over just the ``WARPGROUPS_PER_HEAD`` warpgroups sharing a head.
    Needed once ``NUM_WARPGROUPS > HEADS_PER_TILE``: a head's ``HD`` channels are then split across more
    than one warpgroup, so its RMSNorm sum -- and the ``||q||`` the fixed softmax
    offset is built from -- stop being warpgroup-local. Both sums ride one barrier.
    At ``WARPGROUPS_PER_HEAD == 1`` this is the identity plus a barrier, and the barrier is
    load-bearing even there: it splits a live range that otherwise spills.
    """
    buf[lane, wg, 0] = v0
    buf[lane, wg, 1] = v1
    cute.arch.sync_threads()
    base = (wg // WARPGROUPS_PER_HEAD) * WARPGROUPS_PER_HEAD
    r0 = buf[lane, base, 0]
    r1 = buf[lane, base, 1]
    for j in range(1, WARPGROUPS_PER_HEAD):
        r0 = r0 + buf[lane, base + j, 0]
        r1 = r1 + buf[lane, base + j, 1]
    return r0, r1


def apply_rope(reg, d_t, d_h, d_w, tbl, li_t, li_h, li_w, c0: int = 0, n: int = HD):
    """In-place absolute 3D RoPE on ``reg``, from the SMEM tables.
    ``li_*`` are positions relative to the origin the tables were built at.
    ``reg[e]`` holds head channel ``c0 + e``, so a warpgroup owning only part of a
    head rotates only the pairs landing in its range -- legal because ``c0``, every
    axis base and every range are even, so no ``(2rp, 2rp+1)`` pair straddles the
    boundary.
    """
    (ct, st_, ch, sh, cw, sw) = tbl
    for base, dim, cos_t, sin_t, li in (
        (0, d_t, ct, st_, li_t),
        (d_t, d_h, ch, sh, li_h),
        (d_t + d_h, d_w, cw, sw, li_w),
    ):
        for rp in range(dim // 2):
            c = base + 2 * rp
            if c < c0 or c + 1 >= c0 + n:
                continue
            ca = cos_t[li, rp]
            sa = sin_t[li, rp]
            e = reg[c - c0]
            o = reg[c - c0 + 1]
            reg[c - c0] = e * ca - o * sa
            reg[c - c0 + 1] = e * sa + o * ca


@cute.jit
def fill_rope_axis(cos_t, sin_t, tidx, origin, span, half: cutlass.Constexpr, inv):
    """Cooperatively tabulate cos/sin for one axis over ``span`` positions."""
    i = tidx
    while i < span * half:
        pos = i // half
        rp = i - pos * half
        ang = ACC(origin + pos) * inv[rp]
        cos_t[pos, rp] = cute.cos(ang)
        sin_t[pos, rp] = cute.sin(ang)
        i += THREADS


# --------------------------------------------------------------------------
# The attention half. Both kernels call exactly this code, so there is one place
# where the window mask, the software pipeline and the softmax live.
# They differ only in where K and V come from:
#   fused   the cache fill projects the slab box into one fused K|V cache, so both
#           sources are that cache, the head stride is the box and a panel row is one
#           cache row (``KVRM = 1``).
#   NA      K and V are the caller's own ``(T*H*W*NH, HD)`` tensors, so the sources
#           are two tensors, consecutive heads are one row apart and a panel row is
#           ``NH`` rows (``KVRM = NH``).
# Both are the same affine map, ``base + nh * hstride + panel_row * KVRM``, in units
# of ``HD``-wide rows. ``KVRM`` is compile-time so the fused path's ``* 1`` folds away
# and its addressing is unchanged.
# --------------------------------------------------------------------------


@cute.jit
def attention_tile(
    frags,
    smem,
    acc_qk,
    acc_o,
    bars,
    m_sm,
    l_sm,
    areg,
    kvfrag,
    kv_src,
    kv_base,
    kv_hstride,
    panel,
    window,
    ids,
    NHG: cutlass.Constexpr,
    KVRM: cutlass.Constexpr,
    PT: cutlass.Constexpr,
    PH: cutlass.Constexpr,
    PW: cutlass.Constexpr,
    kt: cutlass.Constexpr,
    kh: cutlass.Constexpr,
    kw: cutlass.Constexpr,
    NKV: cutlass.Constexpr,
):
    """Windowed NA over one query tile's halo panel, for ``NHG`` heads.
    Args:
        frags: ``(frQ, frK, frP, frV)`` -- the four MMA operand fragments.
        smem: ``(sK, sV, sP)`` -- K and V double-buffered, P single.
        acc_qk: two TMEM Q@K accumulators; the pipeline alternates between them.
        acc_o: ``NHG`` TMEM attention-output accumulators, live for the whole loop.
        bars: ``(qkbar, pvbar)`` MMA completion barriers, re-initialised per tile.
        m_sm: ``[lane, nh]`` fixed softmax offsets, from the caller's ``||q||`` pass.
        l_sm: ``[lane, wg, nh]`` softmax denominators, accumulated per warpgroup.
        areg: register scratch for one warpgroup's score row.
        kvfrag: register scratch for one K or V row.
        kv_src: ``(k_src, v_src)``; the fused kernel passes its cache twice.
        kv_base: ``(k_base, v_base)`` -- head 0's first row in each source.
        kv_hstride: rows between consecutive heads, in ``HD``-wide rows.
        panel: ``(t_s, h_s, w_s, pt0, ph0, pw0, bh2, bw2)`` -- the tile's halo panel
            origin, the source box's origin, and that box's H and W extents.
        window: ``(lt0, lh0, lw0)`` -- this query's clamped NA window, panel-relative.
        ids: ``(tidx, lane, wg, warp_i)``.
        NHG: heads processed per call; must be even (the pipeline's buffer parity).
        KVRM: rows between consecutive panel positions in the K/V source.
        PT, PH, PW: halo panel extents.
        kt, kh, kw: NA window extents.
        NKV: KV tiles spanning the panel, ``ceil(PT*PH*PW / M)``.
    On return ``acc_o[nh]`` holds the unnormalised attention output and
    ``l_sm[lane, wg, nh]`` this warpgroup's share of the denominator;
    :func:`attn_epilogue` divides them.
    A two-deep software pipeline over the ``NKV * NHG`` steps. Step ``s`` waits only
    for *its own* Q@K, issued a whole step earlier, and before returning it issues
    P@V(s) and Q@K(s+1) and stages step s+1's K/V row into the other SMEM buffer. So
    a step's source read, its P@V and the next step's Q@K all run underneath the
    current softmax, and only the softmax is on the critical path. Both buffer
    indices are ``nh % 2``, compile-time because ``NHG`` is even, which keeps every
    SMEM slice statically aligned and every barrier phase constant.
    K and V are staged by warpgroups 0 and 1 respectively -- two arms of the same
    ``wg`` test, which costs nothing (the arms already diverge at the store, K being
    K-major and V MN-major) and lets the NA caller pass its own two tensors. A
    *device-value* ``if`` cannot select a tensor: ``v if wg == 1 else k`` fails in the
    NVVM backend, not the frontend.
    """
    frQ, frK, frP, frV = frags
    sK, sV, sP = smem
    qkbar, pvbar = bars
    k_src, v_src = kv_src
    k_base, v_base = kv_base
    lt0, lh0, lw0 = window
    _tidx, lane, wg, warp_i = ids

    N_KEYS = PT * PH * PW  # keys in the halo panel
    PANEL_HW = PH * PW
    NBW = ACC_COLS // 32  # mask words covering one warpgroup's score columns
    RPW = 31 // PW + 2  # panel dw-rows a 32-key mask word can touch

    # Re-initialised per query tile: each barrier then sees exactly ``2 * NKV`` waits
    # in the loop (plus one after it for barrier 0, which retires the pipeline's
    # trailing Q@K), so the phase a given ``nh`` waits on never depends on ``kv``.
    if warp_i == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(qkbar.iterator, 1)
            cute.arch.mbarrier_init(qkbar.iterator + 1, 1)
            cute.arch.mbarrier_init(pvbar.iterator, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    key_row = panel_key_row(0, lane, panel, N_KEYS, PANEL_HW, PW)
    # Prologue: fill buffer 0 with step (0, 0), leave step (0, 1) in registers, and
    # put Q@K(0, 0) in flight.
    if wg == 0:
        _load_head_row(k_src, k_base + key_row * KVRM, kvfrag)
        _store_k_row(kvfrag, sK, lane, 0)
        _load_head_row(k_src, k_base + kv_hstride + key_row * KVRM, kvfrag)
    elif wg == 1:
        _load_head_row(v_src, v_base + key_row * KVRM, kvfrag)
        _store_v_row(kvfrag, sV, lane, 0)
        _load_head_row(v_src, v_base + kv_hstride + key_row * KVRM, kvfrag)
    _publish_smem()
    _mma_issue("n128", frQ, frK, acc_qk[0], 1, 0, 0, False, qkbar.iterator)
    # Prime the P@V barrier: step 0's V store has nothing to wait for, but it still
    # consumes a phase, so the pipeline owes it an arrival.
    if warp_i == 0:
        with cute.arch.elect_one():
            tcgen05.commit(pvbar.iterator)

    kv = 0
    while kv < NKV:
        # Mask for this KV tile: one bit per key, shared by all heads, built per panel
        # *row* rather than per key. The tile's ``M`` keys are a contiguous run of the
        # panel's flattened (dt, dh, dw), so each 32-bit word spans at most ``RPW``
        # dw-rows; every row contributes the same ``kw``-wide run of bits, shifted to
        # wherever that row starts in the word, and only when its (dt, dh) is inside
        # this query's window. Three rows per word against 32 keys per word. Only the
        # words covering this warpgroup's score columns are built.
        wrun = cutlass.Int32((1 << kw) - 1) << lw0
        mask = [cutlass.Int32(0) for _ in range(NBW)]
        for bb in cutlass.range_constexpr(NBW):
            b = wg * NBW + bb
            j0 = kv * M + b * 32
            r0 = j0 // PW
            bits = cutlass.Int32(0)
            for u in cutlass.range_constexpr(RPW):
                r = r0 + u
                rdt = r // PH
                rdh = r - rdt * PH
                if r < PT * PH and rdt >= lt0 and rdt < lt0 + kt and rdh >= lh0 and rdh < lh0 + kh:
                    # Where this row's dw=0 sits in the word. Negative for a row that
                    # started in the previous word; past 31 for one spilling into the
                    # next. Both shifts are guarded -- a shift wider than the type is
                    # poison in the DSL's IR, not a defined zero.
                    off = r * PW - j0
                    if off >= 0:
                        if off < 32:
                            bits = bits | (wrun << off)
                    elif off > -32:
                        bits = bits | (wrun >> (0 - off))
            mask[bb] = bits

        # Whatever filled the K/V source wrote every panel token exactly once, so this
        # is a straight gather -- no per-query projection, norm or RoPE.
        next_key_row = panel_key_row(kv + 1, lane, panel, N_KEYS, PANEL_HW, PW)
        for nh in cutlass.range_constexpr(NHG):
            # Buffer this step owns, and the one the next step will own. ``kv`` is a
            # runtime value -- the KV loop is a real device loop -- so the buffer
            # index has to come from ``nh`` alone, which is what keeps every SMEM
            # slice statically aligned. ``NHG`` is even, so ``nh % 2`` is the flat
            # step's parity.
            bf = nh % 2
            nf = 1 - bf
            # Wait phases. ``pvbar`` is waited once per step, so its phase is the step
            # parity. ``qkbar[bf]`` is waited every second step, so its phase advances
            # by ``NHG / 2`` per KV tile: a constant when that is even, one runtime
            # AND when it is not.
            qk_ph = (cutlass.Int32(kv) * (NHG // 2) + nh // 2) % 2
            # Stage the *next* step's K, start the fetch two steps out, then issue the
            # next step's Q@K -- all before touching this step's scores. That ordering
            # is the point: Q@K(s+1) enters the MMA queue one full softmax before step
            # s+1 waits on it, and the global load has a full step to land. Issuing it
            # after the softmax measures within noise of no pipeline at all.
            # sK[nf] currently holds K(s-1), whose Q@K was waited on last step, so it
            # is free; sV[nf] is not, which is why V is staged further down, behind
            # its own P@V barrier.
            if wg == 0:
                _store_k_row(kvfrag, sK, lane, nf)
                if cutlass.const_expr(nh + 2 < NHG):
                    _load_head_row(k_src, k_base + (nh + 2) * kv_hstride + key_row * KVRM, kvfrag)
                else:
                    _load_head_row(k_src, k_base + (nh + 2 - NHG) * kv_hstride + next_key_row * KVRM, kvfrag)
            _publish_smem()
            _mma_issue("n128", frQ, frK, acc_qk[nf], 1, (nh + 1) % NHG, nf, False, qkbar.iterator + nf)
            # Now wait for *this* step's Q@K, issued one step ago.
            cute.arch.mbarrier_wait(qkbar.iterator + bf, qk_ph)
            # P@V(s-1) has drained, which frees both sP and the V buffer this step is
            # about to stage into. The wait sits here rather than after the
            # exponentials, where it would have more slack: moving it down measures
            # worse, because the V store has to follow it and the global load that
            # follows the store then loses a whole exp loop of latency hiding.
            cute.arch.mbarrier_wait(pvbar.iterator, bf)
            if wg == 1:
                _store_v_row(kvfrag, sV, lane, 2 * nf)
                if cutlass.const_expr(nh + 2 < NHG):
                    _load_head_row(v_src, v_base + (nh + 2) * kv_hstride + key_row * KVRM, kvfrag)
                else:
                    _load_head_row(v_src, v_base + (nh + 2 - NHG) * kv_hstride + next_key_row * KVRM, kvfrag)
            # Softmax over this warpgroup's ``ACC_COLS`` score columns against a *fixed*
            # row offset (the caller put ``||q|| * max||k||`` in ``m_sm``). Nothing
            # here crosses the warpgroup boundary: no max, no max reduction, no
            # rescale of the O accumulator, no denominator reduction. That deletes two
            # CTA-wide barriers and a TMEM read-modify-write from every one of the
            # ``NKV * NHG`` steps.
            m_new = m_sm[lane, nh]
            # One TMEM read. The chunk index may be a runtime value here because a
            # TMEM address is a register either way; only SMEM slices need constexpr.
            _tmem_load(acc_qk[bf], lane, areg, wg)
            cute.arch.fence_view_async_tmem_load()
            # sP still has to be written for a fully masked group (P@V reads all ``M``
            # columns), but zeroing is far cheaper than 32 exps.
            # The group is 32 keys because **only a warp-uniform skip pays**: a warp
            # shares ``qt``, so a group whose panel rows all sit outside the query's dt
            # window is skipped by the whole warp. Finer groups skip more often on
            # paper, but ``lw0`` varies across a warp's lanes, so both arms issue and
            # only the extra tests are real -- measured 9% slower per nibble.
            sm4 = [ACC(0.0) for _ in range(4)]
            for hb in cutlass.range_constexpr(NBW):
                bw = mask[hb]
                if bw != 0:
                    for e in cutlass.range_constexpr(32):
                        # Masking the *score* rather than the exponential keeps the ex2
                        # and the running sum unconditional, so the sum splits across
                        # four accumulators too. ex2 of -1e30 underflows to +0, which
                        # is the point.
                        v = areg[hb * 32 + e] * LOG2E - m_new
                        if (bw >> e) & 1 == 0:
                            v = ACC(-1.0e30)
                        p = cute.math.exp2(v, fastmath=True)
                        areg[hb * 32 + e] = p
                        sm4[e % 4] = sm4[e % 4] + p
                else:
                    for e in cutlass.range_constexpr(32):
                        areg[hb * 32 + e] = ACC(0.0)
            l_sm[lane, wg, nh] = l_sm[lane, wg, nh] + ((sm4[0] + sm4[1]) + (sm4[2] + sm4[3]))
            for g in cutlass.range_constexpr(NUM_WARPGROUPS):
                if wg == g:
                    _fill_a_chunk(sP, lane, areg, g * ACC_COLS, ACC_COLS)
            # Publishes sP for the P@V below and the V buffer for the next step's.
            _publish_smem()
            _mma_issue("pv", frP, frV, acc_o[nh], 2, 0, 2 * bf, True, pvbar.iterator)
        key_row = next_key_row
        kv += 1

    # Retire the pipeline. P@V is the last MMA of the last step, so its barrier proves
    # every accumulator is final -- and, being issued after it, that the trailing
    # Q@K's commit has landed too, which is what makes re-initialising the barriers
    # next tile safe.
    cute.arch.mbarrier_wait(pvbar.iterator, (NKV * NHG) % 2)


@cute.jit
def attn_epilogue(acc_o, l_sm, oreg, sO, lane, wg, NHG: cutlass.Constexpr):
    """Divide the attention accumulators by the denominator, into ``sO``.
    ``sO`` is an A-operand-shaped SMEM buffer whose channel ``nh * HD + d`` is head
    ``nh``'s output: the fused kernel hands it ``sA`` and feeds the out-projection
    straight from it, the NA kernel hands it a buffer it then scatters to global.
    """
    for nh in cutlass.range_constexpr(NHG):
        # Each warpgroup summed its own slice of every score row.
        lden = l_sm[lane, 0, nh]
        for g in cutlass.range_constexpr(1, NUM_WARPGROUPS):
            lden = lden + l_sm[lane, g, nh]
        linv = ACC(0.0)
        if lden > ACC(0.0):
            linv = cute.arch.rcp_approx(lden)
        for g in cutlass.range_constexpr(NUM_WARPGROUPS):
            if wg == g:
                _tmem_load(acc_o[nh], lane, oreg, g, HEAD_COLS)
                for d in cutlass.range_constexpr(HEAD_COLS):
                    oreg[d] = oreg[d] * linv
                _fill_a_chunk(sO, lane, oreg, nh * HD + g * HEAD_COLS, HEAD_COLS)


@cute.jit
def attn_zero_accumulators(acc_o, l_sm, oreg, lane, wg, NHG: cutlass.Constexpr):
    """Clear the flash-attention accumulators (they live in TMEM all loop)."""
    for d in cutlass.range_constexpr(HEAD_COLS):
        oreg[d] = ACC(0.0)
    for nh in cutlass.range_constexpr(NHG):
        for g in cutlass.range_constexpr(NUM_WARPGROUPS):
            if wg == g:
                _tmem_store(acc_o[nh], lane, oreg, g, HEAD_COLS)
        l_sm[lane, wg, nh] = ACC(0.0)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.sync_threads()
