# Working notes for Claude

Context that is expensive to rediscover. The README documents the project for
users; this file records *why* things are the way they are, and which
assumptions have already been tested and found wrong.

## Structure

- `ltx_engine.py` — the whole generation pipeline. No UI. Both front-ends import it.
- `generate_video.py` — Tk GUI only.
- `cli_gen_vid.py` — headless; deliberately has no generation logic of its own,
  it hands stub widget objects to `generation_worker()`.
- `server.py` + `static/index.html` — FastAPI front-end. **Untracked by design**
  (`.gitignore` whitelists specific root files and this is not one), and still
  text-to-video only: none of image/FLF2V/v2v/IC-LoRA is exposed in its UI,
  though `/api/generate` passes `mode` through fine.

Keep it that way. The engine must not import tkinter.

**Anything the front-ends both need goes in the engine, not in each of them.**
There are now three front-ends and they have drifted before — the pre-flight
VRAM estimate once hardcoded "×2 if CFG", which under-counted STG-only and
CFG+STG runs in one place while the engine counted correctly in another.
`effective_tokens()`, `resolve_modality_scale()`, `estimate_vram_gb()` and
`apply_recommended_defaults()` all exist for that reason.

## Generation modes

`config["mode"]` is one of: `text2video`, `image2video`, `flf2v` (start+end
frame), `video2video` (the crossfade — see the finding below), `ic_v2v`
(IC-LoRA, the real video-to-video). Adding one means touching the engine's
mode flags, input loading, stage-1 dispatch, `mode_label`/`mode_tag`, plus the
GUI radio/pickers/validation/config-save, the CLI flag/resolve/summary/
validation, and `server.py`'s DEFAULTS. Grep an existing mode name to find them
all — `flf2v` is the cleanest one to copy.

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

**`video_strength` is a crossfade that compounds, not a restyle dial.** The
narrow useful range is not a tuning problem — it is arithmetic. The condition
is applied at `index=0` and spans the whole clip, so every token gets the same
mask (verified: per-frame correlation to source is flat across all 49 frames,
slope -0.00016/frame), and the denoise loop re-blends
`x0 = model*(1-s) + source*s` on **every** step. The model's free contribution
therefore decays as `(1-s)^num_steps`. Fixed-seed sweep, correlation *between*
outputs: 0.4 vs 1.0 = 0.984, 0.6 vs 1.0 = 0.994, 0.8 vs 1.0 = 0.997, while 0.0
vs any s≥0.2 is ~0.16. A near-step function. Corollary: the range narrows
further as step count rises. Real v2v is IC-LoRA (below), not this.

**IC-LoRA is the actual video-to-video, and the plumbing all works.** Upstream
appends the reference as extra clean tokens the transformer attends to
(`VideoConditionByReferenceLatent`), never touching the generation canvas —
that is why their `conditioning_attention_strength` has a usable middle and
`video_strength` does not. Measured against `LTX-2.3-22b-IC-LoRA-Union-Control`
on this 2.5 fp8 stack: diffusers already converts Lightricks' key format (all
960 tensors, no shim), every targeted module exists in the 2.5 transformer,
blocks 0-47 align, all 480 A/B pairs are shape-clean, and all 960 adapter
tensors arrive in **fp8** — so the existing fp8→bf16 cast is required for
IC-LoRAs too. Two things diffusers will not do for you: it rejects
`duration_head` in `LTX2InContextPipeline(**pipe.components)` (drop that one
key), and it never reads `reference_downscale_factor` from the adapter's
metadata despite taking it as an argument — get that wrong and every reference
token's position is misplaced.

**Reference tokens are paid for out of the same attention budget.** They are
appended to the sequence, so an IC-LoRA run costs `1 + 1/factor²` times the
target's tokens. Anything computing effective tokens must account for it or it
under-counts by 25% (factor 2) to 100% (factor 1) — including the VRAM
calibration, which would otherwise record a token figure the run never had.
`effective_tokens()` is the single place that combines final-stage tokens,
guidance passes and this factor; use it rather than re-deriving.

**The shipped VRAM fit was over-cautious on its own fit machine.** At 6,630
tokens it predicted 8.88GB against 7.87GB actual (13% high). Runs now
self-calibrate into `vram_calibration.json` and the refit sits within ±0.14GB
across 448-12,480 tokens. Two things that matter if you touch this: it must be
keyed by offload profile (a full-resident card holds the ~18GB transformer, so
its intercept is different by more than the entire token term), and it must
refuse to fit from draft-sized runs only — a clean line from three 256×256
renders extrapolated 10x is exactly how you get a confidently wrong ceiling.

**Modality guidance is not tied to CFG.** diffusers gates it on
`modality_scale > 1.0` alone, the same shape as STG's gate, so it rides the
8-step distilled schedule for one extra pass — it never needed CFG's 30 doubled
steps. What it changes, same seed: video correlation 0.93 (barely), audio 0.20
(almost entirely). It is an audio/AV-coupling knob. Caveat: upstream's
`DistilledPipeline` uses `SimpleDenoiser`, "Single transformer call, no
guidance", so 3.0 is their *guided*-preset value and running it on the
distilled schedule is off-recipe — same category as STG-on-distilled, and
resting on one informal A/B.

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

**Dual-boot layout, for the planned WSL2 port.** This box has three NVMes:
`nvme0n1` (this native Linux install, ext4, `/`), `nvme1n1` (Windows boot,
NTFS), `nvme2n1p2` (shared exFAT, labeled "Data Only", not mounted from
Linux by default). exFAT is the intentional choice for the shared drive --
both OSes read/write it natively; NTFS-from-Linux and ext4-from-Windows are
both worse options.

When setting up the WSL2 environment: clone the repo into WSL2's native
ext4 filesystem, not onto the exFAT drive (`/mnt/<drive>/...`) -- exFAT has
no Unix permissions and no real symlink support, and `live-ver` (see below)
is built entirely out of symlinks. Model checkpoints are large enough to be
worth sharing rather than re-downloading -- put them on the exFAT drive
once and symlink to them from WSL2's ext4; that's fine because only the
symlink's *target* crosses onto exFAT, not the symlink itself.

GPU access under WSL2 is a different mechanism from this native install,
not a repeat of it: WSL2 doesn't load the Linux `amdgpu` kernel driver at
all, it goes through the *Windows* GPU driver via paravirtualization. So
the Windows-side driver needs WSL/ROCm compute support installed, not a
second native ROCm kernel-module install inside WSL2. Checked (2026-08-22):
gfx1201 (RX 9070 XT) is officially supported since ROCm 7.2 (March 2026) --
that's also the release that unified Windows/WSL2 into one first-class
target instead of a workaround. Requires Windows 11 25H2 and AMD GPU driver
31.40.1+.

`LinuxHardwareMonitor` (`/sys/class/drm`, `/proc/stat`, `/proc/meminfo`)
will need a rewrite for native Windows but should keep working unmodified
under WSL2, since WSL2 provides its own Linux-style `/proc` and `/sys`.
Untested which of the AMD sysfs GPU attributes (`gpu_busy_percent`,
`mem_info_vram_used`) WSL2 actually populates versus leaves absent.

**IC-LoRAs are indistinguishable from plain LoRAs by inspection.** Same key
layout, same target modules, same rank -- nothing in the tensors says which is
which. `looks_like_ic_lora()` is therefore a heuristic and labelled as one:
`reference_downscale_factor` in the safetensors metadata (reliable when
present, but an adapter trained at factor 1 may omit it), else an `ic-lora`
filename. It only ever warns. Do not promote it to a gate.

**The LoRA stack is part of the pipeline cache identity, scales included.**
Adapters are merged into the resident transformer, so adding, removing,
reordering *or re-scaling* any of them must force a rebuild -- a scale change
alters the weights as much as swapping the file. `build_opts` hashes the whole
ordered list of (path, scale) pairs for that reason.

**The GUI has three separate `✕ Clear` buttons** (positive prompt, negative
prompt, LoRA stack). Any headless test that finds widgets by label text will
grab the wrong one -- this produced a convincing false bug report. Scope the
search to the containing frame instead.

## Conventions

Measure before claiming. Several confident conclusions this project were wrong
until benchmarked — proposing larger VAE tiles (26x worse), attributing decode
time by subtraction, and "fixing" a warning threshold from a contaminated
reading. Findings belong in the README or a code comment, not only in a commit
message.

State what is untested. This runs on exactly one GPU (gfx1201) and the VRAM
model is fitted from that one card.
