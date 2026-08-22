
# 🎬 LTX-2.5 ROCm Video Generator

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/S6I125FYNB)

*An unashamed joint programming collaboration between a human and AI.*\
I started this project with my VERY limited programming skills, and help from Qwen3-Coder:30B. It was a great help and got me to a "working" state. There's no way I would have got there without it.\
I actually got some sleep one night, and in a fit of new-found inspiration I paid up for Claude Pro. I threw my code at it and quickly realised I had been pretty damn lucky to have anything working at all! The way I was instructing Qwen, and my own poor decisions, backed by lack of development knowledge, had me in a bit of a hole. Claude was a breath of fresh air, and I'm now fully embracing the "vibe coding" scene.\
Speed of development has gone through the roof and provided me a "never-quite-final" product that doesn't frustrate me nearly as much as my own work. I know my limits.\
Anyway, hope this is useful for at least one person. And any bugs are entirely Claude's and not mine ;-)

## What is this?!

A GUI control panel and a headless CLI for running the **LTX-2.5 video diffusion model** on a single AMD consumer GPU via ROCm — developed on an RX 9070 XT (16GB VRAM) with 128GB system RAM (originally 32GB, but kept running out by about 2GB!), but the RAM figure is not a requirement — more than 32GB is what matters, and 32GB itself works for smaller generations.

It keeps the FP8 model resident in memory between generations, runs the distilled guidance-free schedule correctly, tiles the VAE decode to avoid AMDGPU driver timeouts, and supports a working two-stage generate-then-upscale pipeline. Is FP8 the correct format? Yet to be seen, but it works for now, and at a decent speed, but I'm very open to constructive criticism!

### Layout

| file | what it is |
|---|---|
| `ltx_engine.py` | The generation pipeline. No UI of any kind. |
| `generate_video.py` | Tk control panel — imports the engine. |
| `cli_gen_vid.py` | Headless CLI — imports the engine. Needs no `tkinter`. |
| `quant_transformer_fp8.py` | One-time FP8 conversion, see [First-time setup](#first-time-setup). |
| `bench_vae_tiles.py` | VAE decode timing harness, see [VAE tile geometry](#vae-tile-geometry--measured-dont-guess). |

Both front-ends share the same `ltx2_config.json` and the same pipeline, so a fix/change in one reaches both.

---

## ✨ Key Features

* **Resident model cache:** the ~18GB FP8 transformer and prompt-embedding cache (in-RAM + on-disk) survive across generations, so only the first click of a session pays the disk-load cost. VRAM is returned automatically after every run, and changing an offload setting rebuilds the pipeline by itself — so on a large-RAM machine you can generally ignore this. An **🧹 Unload Models** button is there for when you want the ~18GB back anyway - disabled during generation. (see [when to use it](#when-to-use-unload-models)).
* **Guidance-free distilled schedule, correctly disabled:** `audio_guidance_scale` defaults to 7.0 in the underlying pipeline, which silently re-enables classifier-free guidance (doubling every step) if you only zero out `guidance_scale`. This script zeroes all guidance/STG scales, so every step actually runs once — roughly 2x fewer FLOPs than a naive guidance-scale-1.0 setup.
* **Real VAE tiling:** spatial tiling (`enable_tiling`) plus framewise (temporal) decoding, which is what actually bounds VRAM and prevents the AMDGPU ring-timeout watchdog from firing on longer clips.
* **Working two-stage upscale (off by default):** generate at a lower base resolution, then run a 2x latent upsample + short refinement pass at the target resolution — faster than generating at full resolution directly, with one noise generator threaded through both stages per the LTX-2.5 reference recipe. Off by default because it pushes VRAM harder — see the hardware warning below before enabling it.
* **Prompt enhancer (optional):** click **✨ Enhance Now** to rewrite a short prompt into the long, detailed caption style LTX-2.5 was trained on, using `google/gemma-4-E2B-it`. The result replaces the prompt box, so you read and edit it before spending minutes on a render — no hidden rewrites. It runs in a throwaway subprocess and frees itself on exit, so it costs **zero resident VRAM or RAM during generation**. Works for image-to-video too (the enhancer is conditioned on your reference frame). Requires the one-off download below.
* **CFG quality mode (off by default):** the distilled schedule is guidance-free, which is fast but drops secondary prompt details. Ticking **CFG quality mode** runs classifier-free guidance on stage 1 — the transformer runs twice per step and pushes away from the negative prompt, which is what forces adherence. Costs roughly **7-8x the compute** (30 steps instead of 8, doubled per step) and about **2x the activation VRAM**, so on 16GB it is realistically single-stage only. Confirmation dialog spells this out before it turns on.
* **STG — structural guidance without a negative prompt:** duplicated limbs, extra fingers, objects floating unattached to anything. **STG** perturbs one transformer block and steers away from the degraded prediction, which targets *structure* rather than prompt adherence — and needs no negative prompt at all. It keeps the 8-step distilled schedule, so it costs one extra pass (**~2x**) rather than CFG mode's 30 doubled steps (~7-8x). Tried here at strength 1.0 and it visibly improved anatomy and object coherence; drop the strength to 0.5 if output looks over-sharpened instead of better formed. Off by default.
* **Live hardware telemetry:** CPU/RAM/GPU/VRAM usage and per-core load, read directly from sysfs/procfs (no extra dependencies).
* **Auto-saved config:** every setting (prompt, resolution, frames, seed, offload tuning) is persisted to `ltx2_config.json` and reloaded on next launch.
* **Cancel support:** stop a run cleanly between diffusion steps.

---

## 🚀 Usage

### First-time setup

Nothing here ships weights — you build the environment and the FP8 model once, locally. Budget ~50GB of disk and an hour, most of it downloading.

0. **Environment.** Python 3.12, and a ROCm build of PyTorch. Install torch from AMD's index *first*, since the PyPI default is a CUDA build and will not work:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install --pre torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/nightly/rocm6.3
   pip install -r requirements.txt
   ```
   Check ROCm actually sees the card before going further — if this prints `False`, nothing below will work:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   `requirements.txt` pins diffusers to a **git commit**, not a release: LTX-2.5 support landed after 0.35, so a PyPI wheel will not do. The GUI additionally needs `tkinter` (`sudo apt install python3-tk` on Debian/Ubuntu); the CLI does not.

1. **Get the LTX-2.5 checkpoint** into `./local_ltx25_model`:
   ```bash
   hf download Lightricks/LTX-2.5-Diffusers --local-dir local_ltx25_model
   ```
   It must be the **`-Diffusers`** repo — `LTX2Pipeline.from_pretrained()` needs the `model_index.json` + per-component subfolder layout that only that one ships. The plain `Lightricks/LTX-2.5` repo is a different format and will not load.

2. **Check `local_ltx25_model/model_index.json`.** If `text_encoder` names `Gemma3ForConditionalGeneration`, change it to `Gemma4UnifiedForConditionalGeneration` — the stock file points at the wrong class for these weights and loading fails with a class/shape mismatch. See `local_ltx25_model/README.md`.

3. **Quantize to FP8** (one time, ~10 min, needs the ~35GB bf16 model in RAM):
   ```bash
   python quant_transformer_fp8.py
   ```
   This casts every `nn.Linear` in the transformer to `float8_e4m3fn` and writes `./local_ltx25_fp8` (~18GB). Keep `./local_ltx25_model` afterwards — the 2-stage upscale path still reads its `latent_upsampler` config.

4. *(Optional)* the prompt enhancer — see [Requirements](#requirements).

### Running

```bash
python generate_video.py
```

This opens the control panel. Fill in:

* **Positive / Negative Prompt** — the negative prompt only takes effect with **CFG quality mode** on; the distilled schedule is guidance-free and never evaluates it. The label above the box says which state you're in. Both boxes have **✕ Clear**.
* **Resolution** — landscape and portrait presets, or *Custom* for any size (snapped to multiples of 32, minimum 256 — the GUI tells you when it adjusts your input). A live readout under the checkbox shows the actual output size and whether the run is single- or 2-stage.
* **2-stage upscale checkbox** — when on, the dropdown is the *base* resolution and the output is 2x it (960x544 → 1920x1088). Off means single-stage at the size shown.
* **Length** — enter as frames or seconds; frame count is auto-aligned to LTX-2.5's `8k + 1` rule.
* **Seed** — a number, or `r` for random.

Click **🚀 Generate Video**. Progress and logs stream into the console pane; **🛑 Cancel** stops between steps.

Tick **🐞 Debug** for per-step timing, latent geometry, token count and a VRAM/RAM reading on every line, plus diffusers' own offload/tiling warnings. It can be toggled mid-run — the next step picks it up, so you never have to restart and reload 18GB to diagnose something.

### Headless CLI

Same engine, no GUI, no `tkinter` required — useful over SSH or in a script:

```bash
python cli_gen_vid.py                              # run what's in the config
python cli_gen_vid.py --prompt "a red car" --seconds 4
python cli_gen_vid.py --dry-run                    # resolve settings, print, exit
python cli_gen_vid.py --image frame.png --debug    # image-to-video
```

Settings come from `ltx2_config.json`; every flag overrides it for that run only. `--save` writes the resolved settings back, `--config other.json` points at a different file.

Unlike the GUI, the VRAM warning is a **hard stop** rather than a prompt — it exits with status 2 and needs `--force` to proceed, which suits unattended runs. `--dry-run` prints the resolved resolution, length, seed, token count and estimated VRAM without generating anything. Ctrl-C sets the same cancel flag the GUI's button does, so it stops cleanly between diffusion steps.

### Recommended settings (16GB VRAM / >32GB RAM)

> ⚠️ **This has only been tested on an RX 9070 XT (16GB VRAM), and headroom on that card is tight.** A 960x544 (2-stage) run at 193 frames — the upper end of what the token-count math below suggested should fit — became unrecoverable during testing here, taking down the desktop session rather than failing cleanly. That's a single observation on one GPU, driver and desktop combination; yours may behave quite differently, better or worse. Treat the figures below as a starting point rather than a validated limit. **2-stage upscaling defaults to off in the GUI** — enable it deliberately, and work up gradually until you know how your own system responds.

The binding constraint is VRAM, not RAM — the transformer's attention cost scales with `latent_frames × (H/32) × (W/32)`, which the script tracks as a "token count". Before generating, it estimates peak VRAM from that count and warns if the estimate is close to your card's capacity (with the option to continue anyway). The threshold is derived from your reported VRAM, so a larger card raises it automatically; see [Tuning knobs](#tuning-knobs-in-ltx2_configjson) to override it. It is a fitted estimate from a single GPU, not a guarantee — don't rely on it as a safety net.

| Mode | Base resolution | Frames | Final output | Notes |
|---|---|---|---|---|
| **Maximum recommended** | 960x544 (2-stage) | up to 121 (5s @24fps) | **1920x1088** | The practical ceiling on 16GB VRAM — don't go higher without headroom to spare |
| Single-stage, no upscale | 1280x704 | up to ~121f | 1280x704 | Skip the checkbox for a direct full-res render |
| ~~Pushing it~~ — not recommended | ~~960x544 (2-stage), up to 193f~~ | | ~~1920x1088~~ | **Failed unrecoverably during testing here.** Untested above the row above; approach with caution on any GPU. |

Going past the maximum recommended row (e.g. full-res 1280x704 direct at 2x, or frame counts above 121 with upscaling on) risks exhausting VRAM or tripping the driver's timeout watchdog. How your system reacts to that — a clean error, a stalled render, or something worse — depends on your driver and desktop. The token-count warning is a fitted guide, not a safety guarantee, so don't rely on it alone.

> ⚠️ **Free up VRAM before a large run: close unneeded GUI programs and run a single monitor.** This is real, not superstition — on this machine the desktop compositor alone (four connected outputs, one active) was holding ~1GB of VRAM at idle before any generation started, and every extra display and GPU-accelerated app (browsers especially — WebGL/video-decode tabs are heavy) adds to that. With only ~1-2GB of headroom above the "maximum recommended" row, that 1GB+ matters. Close browsers/other GPU-heavy apps and disable extra monitors before pushing toward the higher end of the table above.

> ⚠️ **At 32GB RAM, stick to smaller generations.** 32GB works, but it is the floor rather than a comfortable margin: resident model caching (~18GB pinned) plus a transient 23GB text-encoder subprocess plus VAE/upsampler staging can exceed it before VRAM becomes the limit, giving a system-RAM OOM rather than a clean VRAM error. At 32GB, keep to shorter clips and the lower resolutions, click **🧹 Unload Models** between runs (the one configuration where that is normal workflow rather than an escape hatch), and have swap available. Above ~48GB the resident cache stops being a constraint and you can work through the table below normally.

### Tuning knobs (in `ltx2_config.json`)

* `blocks_per_group` — offload group size for the transformer's 48 blocks (default 4 → 12 groups). Raise to 6–8 for more speed at the cost of steady-state VRAM; drop to 2 if the upscale/refinement stage runs out of memory.
* `attention_backend` — defaults to `native` (PyTorch SDPA auto-dispatch, which picks the right kernel for gfx1201 automatically). Only override this if you're deliberately experimenting — a forced kernel can hard-fail on masked cross-attention.

* `vae_tile_size` / `vae_tile_stride` (default 512 / 448) and `vae_tile_frames` / `vae_tile_stride_frames` (default 24 / 16) — VAE decode tile geometry. **The defaults are measured optimal; leave them alone unless you need the low-VRAM fallback below.**
* `stg_mode` / `stg_scale` (default off / 1.0) — Spatio-Temporal Guidance. The strength has a GUI control beside its checkbox, unlike `cfg_scale`, because it is the setting you actually iterate on and editing the config would mean a restart plus an 18GB reload each try.
* `token_warn_threshold` — latent-token count above which the GUI warns before generating. Unset by default, in which case it is computed from your card's reported VRAM using a fitted model (`VRAM_GB ≈ 7.68 + 1.814e-4 × tokens`, with 15% headroom) — roughly 32k tokens on a 16GB card, 70k on 24GB. That fit comes from **one GPU and one run**, so if it proves too cautious or too permissive on your hardware, set an explicit number here to override it.

Both `blocks_per_group` and `attention_backend` are baked into the pipeline when it is built, so they're part of the resident cache's identity: **edit either one and the next run rebuilds automatically** — no restart, no manual unload. (Everything else in `ltx2_config.json` is written by the GUI and read fresh each run.)

> ⚠️ `blocks_per_group` is currently **inert**: `use_stream=True` forces diffusers to a group size of 1 (it logs *"Using streams is only supported for num_blocks_per_group=1"*). That is the faster arrangement anyway — the weight transfer overlaps compute instead of stalling it — so there is no GUI control for it. It only takes effect if you also set `use_stream=False` in the source.

#### VAE tile geometry — measured, don't guess

Decoding 1536x1024 x 49 frames on an RX 9070 XT, timing only the VAE (`bench_vae_tiles.py`):

| spatial | temporal | tile decodes | time | peak VRAM |
|---|---|---|---|---|
| 256px | 24f | 105 | 112.0s | **3.26GB** |
| **512px** | **24f** | **36** | **24.8s** | 6.63GB |
| 512px | 48f | 24 | 225.9s | 12.42GB |
| 1024px | 24f | 6 | 653.1s | 15.03GB |
| 1024px | 48f | — | **OOM** | — |

Chart: [Tile Size vs Decode Time](https://claude.ai/code/artifact/0a1fa72d-bd59-4a47-a4dc-0227ea28bccf) — the U-shaped curve across the 256/512/1024px row at fixed 24f.

The defaults (512/24) win by 4.5x over smaller tiles and 26x over larger ones. Bigger tiles are the *worse* direction on both axes: peak VRAM tracks the largest single tile's working set rather than the tile count, and once a tile stops fitting the allocator thrashes far more than the saved overlap-blending is worth. 1024px peaks at 15.03GB of 15.9GB — close enough to capacity to be risky.

**Strides must land on a latent boundary** (spatial compression is 32, temporal is 8). A ragged stride is punished hard: a 384px tile with a 336px stride (10.5 latent px) ran 3x slower than a 512px tile *despite decoding fewer tiles*. Keep `vae_tile_size` a multiple of 32 and `vae_tile_frames` a multiple of 8, with strides likewise.

**Low-VRAM fallback:** `vae_tile_size: 256`, `vae_tile_stride: 224` halves decode VRAM (6.63 → 3.26GB) for ~4.5x the decode time. Worth it only if you want to push resolution or frame count past what currently fits.

---

## 🛠️ Troubleshooting

### AMDGPU Ring Timeout (Black screen / Session Crash)
If your display manager crashes to the login screen during VAE decode, the kernel's DRM watchdog killed the GPU process because a compute kernel ran too long. VAE tiling in this script should prevent this at the recommended settings above, but if you still hit it, increase the driver's timeout limits:

```bash
sudo bash -c 'cat << EOF > /etc/modprobe.d/amdgpu-timeout.conf
options amdgpu gpu_recovery=1
options amdgpu lockup_timeout=60000,60000,60000,60000
EOF'
sudo update-initramfs -u
```
*(Requires a reboot.)*

### Slow VAE decode? Check `MIOPEN_FIND_MODE`

This script used to force `MIOPEN_FIND_MODE=1` (NORMAL = exhaustive kernel search). On an RX 9070 XT that cost **13.8x on the VAE decode**:

| `MIOPEN_FIND_MODE` | vae.decode (1536x1024, 49f) | peak VRAM |
|---|---|---|
| `1` (exhaustive) | 342.5s | 6.63GB |
| unset (MIOpen heuristic) | **24.8s** | 6.63GB |

Identical shape, tiles and peak VRAM — the same computation, just a different kernel-selection strategy. The penalty **recurs on every run**: exhaustive mode re-searches even once MIOpen's perf database is populated, so it is not a one-off warm-up cost. It is now left unset; export it yourself only if you have a specific reason to force a mode.

On a 2-stage 97-frame run this was roughly six minutes of the ten.

### Out of Memory (VRAM) Errors
1. Drop `blocks_per_group` to 2 in `ltx2_config.json`. This takes effect on the next run on its own — the pipeline rebuilds because the setting changed.
2. Reduce resolution or frame count (see table above).
3. Close GPU-heavy apps and extra monitors (see the VRAM warning above — the compositor alone can hold ~1GB).

<a id="when-to-use-unload-models"></a>
### When to use 🧹 Unload Models

Less often than you'd think. VRAM is already released after every run — the generation path runs `gc.collect()` + `empty_cache()` in a `finally` block, success or failure — and the two build-time knobs above now rebuild the pipeline by themselves when changed. So on a large-RAM machine it is an escape hatch, not part of the loop.

Reach for it when:

* **You're at or near 32GB RAM** — here it *is* routine. ~18GB pinned plus a transient 23GB text-encoder subprocess doesn't leave room to keep models resident between runs.
* **You want the GPU and RAM back for something else** — a game, another ML job — without closing the control panel.
* **Something went wrong** and you'd rather start the next attempt from a clean pipeline than debug a half-torn-down one.

The cost is one slow run afterwards: the next generation re-reads ~18GB from disk. The log line tells you how much it actually freed.

Peak RAM usage is roughly 45GB — the resident 18GB transformer plus a transient text-encoder subprocess. **Comfortably above that (~48GB+) and RAM stops being a factor**, so swap is optional insurance. At 32GB that 45GB peak won't fit in RAM alone, so a swap file (16GB+) is required rather than optional, and you should keep to smaller generations — see the warning above.

---

## Requirements

* **AMD GPU with ROCm support, bfloat16, and enough VRAM.** Developed and tested on exactly one card — an RX 9070 XT (gfx1201). Nothing here is architecture-specific, so other ROCm-capable AMD GPUs *should* work, but that is untested and not a promise. Two things decide it:

  * **bfloat16 is a hard requirement** — the FP8 weights are upcast to bf16 in the patched linear layers, and the rest of the pipeline runs in bf16 throughout. Well supported on RDNA3/4 and CDNA; weak or emulated on older architectures.
  * **VRAM is the binding constraint.** The fixed footprint is ~8GB before any activations, and the rest scales with `latent_frames × (H/32) × (W/32)`:

  | VRAM | usable latent tokens | in practice |
  |---|---|---|
  | 8GB | ~0 | won't fit |
  | 12GB | ~14,000 | short clips, low resolution only |
  | **16GB** | ~32,000 | the tested configuration |
  | 24GB | ~70,000 | comfortable; 2-stage + CFG together become viable |

  Those figures come from a model fitted on one GPU (see [Tuning knobs](#tuning-knobs-in-ltx2_configjson)), so treat them as a guide for judging your own card rather than a specification. The GUI computes the same numbers at runtime from your reported VRAM and warns before a run that looks too large.
* **System RAM: more than 32GB recommended; 32GB is the working minimum.** Peak usage is ~45GB, so anything comfortably above that (~48GB+) removes RAM as a constraint. 32GB works for smaller generations provided you have swap and use **🧹 Unload Models** between runs — see the RAM warning under Recommended Settings. Development happened on 128GB, but that is not a requirement.
* Pre-quantized FP8 weights in `./local_ltx25_fp8` — built once by `quant_transformer_fp8.py`, see [First-time setup](#first-time-setup)
* The base LTX-2.5 model directory (`./local_ltx25_model`) for the latent upsampler config used by the 2-stage upscale path
* *(Optional, for the prompt enhancer)* `google/gemma-4-E2B-it` in `./local_ltx25_enhancer` — ungated, Apache-2.0, ~10GB, no HF token needed. The LTX-2.5 checkpoints ship `prompt_enhancer` and `processor` as nulls, so this is a separate download:
  ```bash
  hf download google/gemma-4-E2B-it --local-dir local_ltx25_enhancer
  ```
### PLEASE FEEL FREE TO CONTRIBUTE TO MY "Buy Euan an RTX 5090 Fund" ;-)
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/S6I125FYNB)
