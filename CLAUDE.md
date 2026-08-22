# Working notes for Claude

Context that is expensive to rediscover. The README documents the project for
users; this file records *why* things are the way they are, and which
assumptions have already been tested and found wrong.

## Structure

- `ltx_engine.py` — the whole generation pipeline. No UI. Both front-ends import it.
- `generate_video.py` — Tk GUI only.
- `cli_gen_vid.py` — headless; deliberately has no generation logic of its own,
  it hands stub widget objects to `generation_worker()`.

Keep it that way. The engine must not import tkinter.

## Findings that cost real measurement

**`MIOPEN_FIND_MODE=1` is a 13.8x trap.** It was set for years "for stability".
Measured: 342.5s vs 24.8s for an identical VAE decode — same shape, same tiles,
same 6.63GB peak. The penalty *recurs every run*; exhaustive mode re-searches
even with MIOpen's perf database populated. Do not re-add it.

**VAE tile defaults (512px / 24 frames) are optimal — measured, not assumed.**
Bigger tiles are dramatically worse (1024px was 26x slower and peaked at
15.03GB), smaller are worse too (256px was 4.5x slower). Peak VRAM tracks the
*largest single tile*, not the tile count. Strides must land on a latent
boundary (spatial 32, temporal 8) — a ragged stride cost 3x. `benchmarks/bench_vae_tiles.py`
reproduces all of this.

**`blocks_per_group` is inert** while `use_stream=True` — diffusers forces the
group size to 1 and logs that it is doing so. That is the faster arrangement
anyway. There is deliberately no GUI control for it.

**`torch.cuda.empty_cache()` frees zero pinned host memory.** Group offload uses
`low_cpu_mem_usage=False`, i.e. page-locked buffers, and the caching host
allocator holds them. `torch._C._host_emptyCache()` is what actually returns
the ~18GB.

**STG works with the distilled schedule.** `stg_scale=1.0` on the 8-step
distilled sigmas visibly improved anatomy (finger count, hands merging) and
object coherence (things floating unattached), despite that schedule being
trained guidance-free. Costs one extra forward pass — 2x — but keeps the 8
steps, so it is ~4x cheaper than CFG mode's 30 doubled steps. It uses **no
negative prompt**: it perturbs transformer block 28 and steers away from the
degraded prediction, so it targets structure, not prompt adherence. That is the
answer to "can I use a negative prompt without CFG" — you can't, but STG is
usually what was actually wanted. Caveat: one comparison, not a controlled
measurement.

**Output filenames encode guidance mode** (`_stg1`, `_cfg3`). Without that a
same-seed A/B writes the same filename twice and the second run silently
destroys the first — which happened before it was fixed.

**Exit segfault.** Letting Python finalise after a GPU session crashes in the
ROCm runtime (`ip == fault address`, i.e. a call into an unloaded library). The
close handler calls `os._exit(0)`. It deliberately does *not* free models first
— the kernel reclaims everything, and doing it by hand only made closing slow.

## Traps

**`mp.set_start_method("spawn")`.** Editing a module while the GUI is running
breaks its *next* subprocess, because the child re-imports from disk and
resolves targets by name. Symptom: "Text encoding failed" plus a traceback
whose line numbers don't match the source. Always restart the GUI after edits.

**`live-ver` drift.** `~/LTX-2.5/live-ver` symlinks the code but keeps its own
`ltx2_config.json`. Adding a new module breaks it until a symlink is added;
config-level fixes must be applied twice.

**sysfs VRAM is system-wide.** The `[dbg]` VRAM figures include every process on
the GPU. A reading of 15.58GB once turned out to include 2.38GB from an
unrelated benchmark and made a healthy run look like a 98% near-miss. Use
`torch.cuda.max_memory_reserved()` (process-local) for anything load-bearing.

## Conventions

Measure before claiming. Several confident conclusions this project were wrong
until benchmarked — proposing larger VAE tiles (26x worse), attributing decode
time by subtraction, and "fixing" a warning threshold from a contaminated
reading. Findings belong in the README or a code comment, not only in a commit
message.

State what is untested. This runs on exactly one GPU (gfx1201) and the VRAM
model is fitted from that one card.
