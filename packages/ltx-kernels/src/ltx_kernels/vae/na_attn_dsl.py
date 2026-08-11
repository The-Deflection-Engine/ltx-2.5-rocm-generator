"""Standalone 3D neighborhood attention -- a drop-in for ``natten.na3d``.
Attention only: Q/K/V arrive already projected, RMS-normed and RoPE'd, so nothing
here scales with the channel count. It runs ``HG`` heads at a time and loops over
head groups, which keeps TMEM and SMEM independent of the head count while still
reading each (KV tile, head) row exactly once. Used for the DiffVAE's deterministic
stages, whose 32/16/8/8 heads the fused stage-5 kernel's TMEM map cannot hold.
The window mask, the two-deep pipeline and the softmax are not reimplemented here --
they are :func:`ltx_kernels.vae.fna_attn_core.attention_tile`, the same trace the
fused kernel runs.
Q must arrive carrying the attention scale, exactly as ``natten.na3d`` expects it.
The one extra thing this kernel needs is the ``k_norm`` weight, from which it builds a
fixed softmax offset in place of an online row max -- see
:mod:`ltx_kernels.vae.softmax_bound`, which is worth reading before calling this.
Compile keys are ``(kernel_size, tile_thw)`` and nothing else -- head count, T, H and
W are all runtime -- so the four deterministic stages share two binaries.
"""

# NOTE: no `from __future__ import annotations` -- cute.struct needs real types.

import os
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.utils import SmemAllocator

from ltx_kernels.vae.availability import UNSUPPORTED_MESSAGE, gpu_supports_dsl_kernels
from ltx_kernels.vae.fna_attn_core import (
    _as_rows,
    _dyn,
    _dyn_act,
    attention_tile,
    attn_epilogue,
    attn_zero_accumulators,
    cta_count,
    default_tile_thw,
)
from ltx_kernels.vae.fna_gemm import UmmaSharedStorage, _mma_op, _reg_shape, acquire_tmem, release_tmem
from ltx_kernels.vae.fna_geometry import (
    halo_panel_origin,
    query_row_indices,
    query_slot_in_tile,
    unflatten_3d,
    window_origin_in_panel,
)
from ltx_kernels.vae.fna_smem import (
    _alias,
    _fill_a_gather,
    _read_a_chunk,
    _scatter_a,
    cooperative_copy_slot,
)
from ltx_kernels.vae.fna_types import (
    ACC,
    HD,
    HEAD_COLS,
    IO,
    LOG2E,
    NUM_WARPGROUPS,
    THREADS,
    TILE_K,
    TILE_N,
    TILER_N64,
    TILER_N128,
    M,
)
from ltx_kernels.vae.softmax_bound import softmax_row_bound

_COMPILED: "dict[tuple, Any]" = {}

# Heads per group. ``HG * HD`` TMEM columns for the attention accumulators plus 256
# for the two Q@K accumulators must fit in 512, so 4 is the ceiling; it also has to
# be even (the pipeline's buffer parity is ``nh % 2``) and to divide ``NUM_WARPGROUPS`` evenly
# for the ``||q||`` pass below. Every DiffVAE stage's head count is a multiple of 4,
# so one value serves all of them and the head count stays a runtime argument.
HG = 4


@cute.kernel
def _na_kernel(
    q: cute.Tensor,  # (T*H*W, NH*HD) bf16, RMS-normed, scaled and RoPE'd
    k: cute.Tensor,  # (T*H*W*NH, HD) bf16, RMS-normed and RoPE'd
    v: cute.Tensor,  # (T*H*W*NH, HD) bf16
    y: cute.Tensor,  # (T*H*W + M, NH*HD) bf16
    k_bound_t: cute.Tensor,  # (1,) f32, see ltx_kernels.vae.softmax_bound
    T,
    H,
    W,
    NH,
    n_hg,
    n_tiles,
    n_ctas,
    grid_hw,
    grid_w_,
    kt: cutlass.Constexpr,
    kh: cutlass.Constexpr,
    kw: cutlass.Constexpr,
    TT: cutlass.Constexpr,
    TH: cutlass.Constexpr,
    TW: cutlass.Constexpr,
    PT: cutlass.Constexpr,
    PH: cutlass.Constexpr,
    PW: cutlass.Constexpr,
    NKV: cutlass.Constexpr,
    mma128: cute.TiledMma,
    mma64: cute.TiledMma,
    p_layout: cute.ComposedLayout,
    q_layout: cute.ComposedLayout,
    k_layout: cute.ComposedLayout,
    v_layout: cute.ComposedLayout,
    o_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    cta_index, _, _ = cute.arch.block_idx()
    lane = tidx % M
    wg = tidx // M
    warp_i = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    volume = (T, H, W)
    radius = ((kt - 1) // 2, (kh - 1) // 2, (kw - 1) // 2)
    panel_shape = (PT, PH, PW)
    kernel_shape = (kt, kh, kw)
    HPG = HG // NUM_WARPGROUPS  # heads whose ||q|| one warpgroup computes

    smem = SmemAllocator()
    storage = smem.allocate(UmmaSharedStorage)
    # One 96 KiB pool: ``sO`` (the epilogue's staging buffer) is dead for the whole
    # KV loop and ``sP``/``sK``/``sV`` are dead outside it, so they alias.
    pool_elems = 6 * M * TILE_K
    pool = smem.allocate_tensor(IO, cute.make_layout((pool_elems,)), byte_alignment=128)
    sO = _alias(pool, o_layout)
    sP = _alias(pool, p_layout)
    sK = _alias(pool, k_layout, 2 * M * TILE_K)
    sV = _alias(pool, v_layout, 4 * M * TILE_K)
    sQ = smem.allocate_tensor(IO, q_layout.outer, byte_alignment=128, swizzle=q_layout.inner)
    qkbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((2,)), byte_alignment=16)
    pvbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), byte_alignment=16)
    # ``rowix`` is always an in-bounds row and feeds the gather; ``rowix_w`` is where
    # a lane's result goes, which for an out-of-range lane of a partial tile is a
    # scratch row past the end of the output.
    rowix = smem.allocate_tensor(cutlass.Int32, cute.make_layout((M,)), byte_alignment=16)
    rowix_w = smem.allocate_tensor(cutlass.Int32, cute.make_layout((M,)), byte_alignment=16)
    m_sm = smem.allocate_tensor(ACC, cute.make_layout((M, HG)), byte_alignment=16)
    l_sm = smem.allocate_tensor(ACC, cute.make_layout((M, NUM_WARPGROUPS, HG)), byte_alignment=16)
    cute.arch.sync_threads()

    tmem, tmem_ptr = acquire_tmem(storage)

    lay_main = mma128.make_fragment_C(mma128.partition_shape_C(TILER_N128[:2])).layout
    lay_n64 = mma64.make_fragment_C(mma64.partition_shape_C(TILER_N64[:2])).layout
    # Same TMEM map as the fused kernel, with ``HG`` standing in for its head count:
    # the per-head attention accumulators low, the two Q@K accumulators above.
    scr = HG * HD
    acc_o = [cute.make_tensor(tmem_ptr + h * HD, lay_n64) for h in range(HG)]
    acc_qk = [cute.make_tensor(tmem_ptr + scr + b * TILE_N, lay_main) for b in range(2)]
    rshape = _reg_shape(mma128, acc_qk[0], lane, TILE_N)
    rshape_o = _reg_shape(mma64, acc_o[0], lane, HD, HEAD_COLS)

    frQ = mma128.make_fragment_A(sQ)
    frK = mma128.make_fragment_B(sK)
    frP = mma64.make_fragment_A(sP)
    frV = mma64.make_fragment_B(sV)

    areg = cute.make_rmem_tensor(rshape, ACC)
    oreg = cute.make_rmem_tensor(rshape_o, ACC)
    kvfrag = cute.make_rmem_tensor(cute.make_layout(HD), IO)

    copy_slot = cooperative_copy_slot(tidx)
    slot_t, slot_h, slot_w = query_slot_in_tile(lane, TH, TW)

    tile_id = cta_index
    while tile_id < n_tiles:
        tile_i, tile_j, tile_k = unflatten_3d(tile_id, grid_hw, grid_w_)
        tile_origin = (tile_i * TT, tile_j * TH, tile_k * TW)

        panel_origin = halo_panel_origin(tile_origin, radius, panel_shape, volume)
        query_pos, read_row, write_row = query_row_indices(
            (tile_origin[0] + slot_t, tile_origin[1] + slot_h, tile_origin[2] + slot_w),
            volume,
            lane,
            T * H * W,
        )
        window = window_origin_in_panel(query_pos, panel_origin, radius, kernel_shape, volume)
        # ``panel_key_row`` indexes the caller's K/V directly, so the box it walks is
        # the whole volume: origin zero, extents H and W.
        panel = (*panel_origin, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), H, W)

        cute.arch.sync_threads()
        rowix[lane] = read_row
        rowix_w[lane] = write_row
        cute.arch.sync_threads()

        hg = 0
        while hg < n_hg:
            c_off = hg * (HG * HD)
            # This group's queries: one coalesced gather straight into the swizzled
            # operand. The channel window moves with the group; the operand does not.
            _fill_a_gather(sQ, q, rowix, copy_slot, HG * HD, c_off)
            cute.arch.sync_threads()
            # ``||q||`` per (row, head) -- the fixed softmax offset. Each lane owns its
            # own row, so a head's channels are register-local and this needs no
            # reduction at all; the warpgroups split the heads.
            for h_loc in cutlass.range_constexpr(HPG):
                for g in cutlass.range_constexpr(NUM_WARPGROUPS):
                    if wg == g:
                        nh = g * HPG + h_loc
                        _read_a_chunk(sQ, lane, areg, nh * HD, HD)
                        s = ACC(0.0)
                        for d in cutlass.range(HD, unroll_full=True):
                            s += areg[d] * areg[d]
                        # Read at the point of use rather than hoisted: this runs
                        # once per head group, and a hoisted value would hold a
                        # register live across the KV loop.
                        m_sm[lane, nh] = cute.sqrt(s) * k_bound_t[0] * LOG2E
            # Ends on a barrier, which is also what publishes ``m_sm``.
            attn_zero_accumulators(acc_o, l_sm, oreg, lane, wg, HG)

            # K and V are the caller's own tensors: consecutive heads are one
            # ``HD``-row apart and one panel row is ``NH`` of them.
            attention_tile(
                (frQ, frK, frP, frV),
                (sK, sV, sP),
                acc_qk,
                acc_o,
                (qkbar, pvbar),
                m_sm,
                l_sm,
                areg,
                kvfrag,
                (k, v),
                (hg * HG, hg * HG),
                cutlass.Int32(1),
                panel,
                window,
                (tidx, lane, wg, warp_i),
                HG,
                NH,
                PT,
                PH,
                PW,
                kt,
                kh,
                kw,
                NKV,
            )

            cute.arch.sync_threads()
            attn_epilogue(acc_o, l_sm, oreg, sO, lane, wg, HG)
            cute.arch.sync_threads()
            _scatter_a(sO, y, rowix_w, copy_slot, HG * HD, c_off)
            cute.arch.sync_threads()
            hg += 1

        tile_id += n_ctas

    release_tmem(tmem, tmem_ptr)


@cute.jit
def _launch(
    q,
    k,
    v,
    y,
    k_bound_t,
    T,
    H,
    W,
    NH,
    n_hg,
    n_tiles,
    n_ctas,
    grid_hw,
    grid_w_,
    kt: cutlass.Constexpr,
    kh: cutlass.Constexpr,
    kw: cutlass.Constexpr,
    TT: cutlass.Constexpr,
    TH: cutlass.Constexpr,
    TW: cutlass.Constexpr,
    PT: cutlass.Constexpr,
    PH: cutlass.Constexpr,
    PW: cutlass.Constexpr,
    NKV: cutlass.Constexpr,
):
    mma128 = cute.make_tiled_mma(_mma_op("n128"))
    mma64 = cute.make_tiled_mma(_mma_op("pv"))
    p_layout = sm100_utils.make_smem_layout_a(mma64, TILER_N64, IO, 2)
    q_layout = sm100_utils.make_smem_layout_a(mma128, TILER_N128, IO, HG)
    k_layout = sm100_utils.make_smem_layout_b(mma128, TILER_N128, IO, 2)
    v_layout = sm100_utils.make_smem_layout_b(mma64, TILER_N64, IO, 4)
    # ``sO`` is an A-operand only so the epilogue can reach it with the same
    # ``_fill_a_chunk`` the fused kernel uses; nothing multiplies by it.
    o_layout = sm100_utils.make_smem_layout_a(mma128, TILER_N128, IO, HG * HD // 64)

    _na_kernel(
        q,
        k,
        v,
        y,
        k_bound_t,
        T,
        H,
        W,
        NH,
        n_hg,
        n_tiles,
        n_ctas,
        grid_hw,
        grid_w_,
        kt,
        kh,
        kw,
        TT,
        TH,
        TW,
        PT,
        PH,
        PW,
        NKV,
        mma128,
        mma64,
        p_layout,
        q_layout,
        k_layout,
        v_layout,
        o_layout,
    ).launch(grid=[n_ctas, 1, 1], block=[THREADS, 1, 1])


def na_attn_available(device_index: int = 0) -> bool:
    """Whether this GPU can run the NA kernel at all (datacenter Blackwell)."""
    return gpu_supports_dsl_kernels(device_index)


def na_supported(
    *,
    num_heads: int,
    head_dim: int,
    kernel_size: "tuple[int, int, int]",
    T: int,
    H: int,
    W: int,
    tile_thw: "tuple[int, int, int] | None" = None,
) -> bool:
    """Whether this shape is served by the standalone NA kernel."""
    if head_dim != HD or num_heads % HG != 0:
        return False
    tt, th, tw = tile_thw or default_tile_thw(kernel_size)
    kt, kh, kw = kernel_size
    pt, ph, pw = tt + kt - 1, th + kh - 1, tw + kw - 1
    if max(pt, ph, pw) > 32:
        return False
    return pt <= T and ph <= H and pw <= W


def _compile(*, kernel_size: "tuple[int, int, int]", tile_thw: "tuple[int, int, int]"):
    key = (kernel_size, tile_thw)
    if key in _COMPILED:
        return _COMPILED[key]
    kt, kh, kw = kernel_size
    tt, th, tw = tile_thw
    pt, ph, pw = tt + kt - 1, th + kh - 1, tw + kw - 1
    # The mask packs each key's panel-local offset into 5-bit fields.
    if max(pt, ph, pw) > 32:
        raise ValueError(f"halo panel {(pt, ph, pw)} exceeds the packed-offset limit of 32")
    nkv = (pt * ph * pw + TILE_N - 1) // TILE_N
    dev = torch.device("cuda")

    def za(a: int, b: int):
        return _dyn_act(torch.zeros(a, b, device=dev, dtype=torch.bfloat16))

    # Pass-through for CuTe DSL compiler flags, e.g. --keep-sass / --ptxas-options=-v.
    opts = os.environ.get("CUTE_DSL_OPTS")
    _COMPILED[key] = cute.compile(
        _launch,
        za(M, HG * HD),
        za(M, HD),
        za(M, HD),
        za(M, HG * HD),
        _dyn(torch.zeros(1, device=dev, dtype=torch.float32)),
        4,
        4,
        4,
        HG,
        1,
        1,
        1,
        1,
        1,
        kt,
        kh,
        kw,
        tt,
        th,
        tw,
        pt,
        ph,
        pw,
        nkv,
        **({"options": opts} if opts else {}),
    )
    return _COMPILED[key]


def run_na_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: "tuple[int, int, int]",
    k_norm_weight: torch.Tensor,
    tile_thw: "tuple[int, int, int] | None" = None,
) -> torch.Tensor:
    """Windowed 3D neighborhood attention over ``(1, T, H, W, NH, HD)`` Q/K/V.
    ``q`` must already carry the attention scale, and ``k`` must be the output of an
    RMSNorm with ``k_norm_weight``. Returns ``(1, T, H, W, NH * HD)``.
    Derives the softmax bound from ``k_norm_weight`` on every call, which is a small
    reduction per call; a caller in a hot loop should hoist it and use
    :func:`run_na_attention_bound`.
    """
    k_bound = softmax_row_bound(k_norm_weight)
    return run_na_attention_bound(q, k, v, kernel_size=kernel_size, k_bound=k_bound, tile_thw=tile_thw)


def run_na_attention_bound(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: "tuple[int, int, int]",
    k_bound: torch.Tensor,
    tile_thw: "tuple[int, int, int] | None" = None,
) -> torch.Tensor:
    """:func:`run_na_attention` with the softmax bound already computed.
    ``k_bound`` is the 0-d tensor :func:`ltx_kernels.vae.softmax_row_bound` returns; it
    is read by the kernel as an operand, so it never leaves the device.
    """
    if not na_attn_available(q.device.index or 0):
        raise RuntimeError(UNSUPPORTED_MESSAGE)
    if q.shape[0] != 1:
        raise ValueError("B=1 only")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"q/k/v shapes differ: {q.shape} {k.shape} {v.shape}")
    _, T, H, W, NH, hd = (int(s) for s in q.shape)
    tt, th, tw = tile_thw or default_tile_thw(kernel_size)
    if not na_supported(num_heads=NH, head_dim=hd, kernel_size=kernel_size, T=T, H=H, W=W, tile_thw=(tt, th, tw)):
        raise ValueError(f"na kernel unsupported: NH={NH} hd={hd} shape={(T, H, W)} k={kernel_size}")

    q_ = _as_rows(q[0].reshape(T, H, W, NH * hd).to(dtype=torch.bfloat16))
    # K and V are addressed one head-row at a time, so they are ``(tokens*NH, HD)``.
    k_ = k[0].reshape(T, H, W, NH * hd).to(dtype=torch.bfloat16).contiguous().view(-1, hd)
    v_ = v[0].reshape(T, H, W, NH * hd).to(dtype=torch.bfloat16).contiguous().view(-1, hd)

    grid_t = (T + tt - 1) // tt
    grid_h = (H + th - 1) // th
    grid_w = (W + tw - 1) // tw
    n_tiles = grid_t * grid_h * grid_w
    n_ctas = cta_count(n_tiles, q_.device)
    # M extra rows absorb the stores of out-of-range lanes in partial tiles.
    y_buf = torch.empty((T * H * W + M, NH * hd), device=q_.device, dtype=torch.bfloat16)

    compiled = _compile(kernel_size=kernel_size, tile_thw=(tt, th, tw))
    compiled(
        _dyn_act(q_),
        _dyn_act(k_),
        _dyn_act(v_),
        _dyn_act(y_buf),
        _dyn(k_bound.detach().reshape(1).float().to(device=q_.device)),
        T,
        H,
        W,
        NH,
        NH // HG,
        n_tiles,
        n_ctas,
        grid_h * grid_w,
        grid_w,
    )
    return y_buf[: T * H * W].view(1, T, H, W, NH * hd).to(dtype=q.dtype)
