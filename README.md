# 🎬 LTX-2.5 ROCm Video Generator

A GUI control panel (`generate_video.py`) for running the **LTX-2.5 video diffusion model** on a single AMD consumer GPU via ROCm — built and tuned on an RX 9070 XT (16GB VRAM) with 124GB system RAM.

It keeps the FP8 model resident in memory between generations, runs the distilled guidance-free schedule correctly, tiles the VAE decode to avoid AMDGPU driver timeouts, and supports a working two-stage generate-then-upscale pipeline.

> `gen_test.py` in this repo is an old prototype kept for reference only — it is **not** maintained and should not be used. `generate_video.py` is the active script.

---

## ✨ Key Features

* **Resident model cache:** the ~18GB FP8 transformer and prompt-embedding cache (in-RAM + on-disk) survive across generations, so only the first click of a session pays the disk-load cost. A **🧹 Free Models** button releases them when you're done.
* **Guidance-free distilled schedule, correctly disabled:** `audio_guidance_scale` defaults to 7.0 in the underlying pipeline, which silently re-enables classifier-free guidance (doubling every step) if you only zero out `guidance_scale`. This script zeroes all guidance/STG scales, so every step actually runs once — roughly 2x fewer FLOPs than a naive guidance-scale-1.0 setup.
* **Real VAE tiling:** spatial tiling (`enable_tiling`) plus framewise (temporal) decoding, which is what actually bounds VRAM and prevents the AMDGPU ring-timeout watchdog from firing on longer clips.
* **Working two-stage upscale (off by default):** generate at a lower base resolution, then run a 2x latent upsample + short refinement pass at the target resolution — faster than generating at full resolution directly, with one noise generator threaded through both stages per the LTX-2.5 reference recipe. Off by default because it pushes VRAM harder — see the hardware warning below before enabling it.
* **Live hardware telemetry:** CPU/RAM/GPU/VRAM usage and per-core load, read directly from sysfs/procfs (no extra dependencies).
* **Auto-saved config:** every setting (prompt, resolution, frames, seed, offload tuning) is persisted to `ltx2_config.json` and reloaded on next launch.
* **Cancel support:** stop a run cleanly between diffusion steps.

---

## 🚀 Usage

```bash
python generate_video.py
```

This opens the control panel. Fill in:

* **Positive / Negative Prompt** — negative prompt is accepted but unused; the distilled schedule runs guidance-free.
* **Resolution** — pick a preset or choose *Custom* and enter width/height directly (must be multiples of 32).
* **2-stage upscale checkbox** — when on, output is 2x the selected resolution (e.g. select 960x544 to get a 1920x1088 final video).
* **Length** — enter as frames or seconds; frame count is auto-aligned to LTX-2.5's `8k + 1` rule.
* **Seed** — a number, or `r` for random.

Click **🚀 Generate Video**. Progress and logs stream into the console pane; **🛑 Cancel** stops between steps.

### Recommended settings for this hardware (16GB VRAM / 124GB RAM)

> ⚠️ **This has only been tested on an RX 9070 XT (16GB VRAM), and headroom on that card is tight.** A 960x544 (2-stage) run at 193 frames — the upper end of what the token-count math below suggested should fit — locked up the entire machine during testing, not just the generation process. Treat the figures below as the practical ceiling, not a floor to push past, until proven otherwise on your own hardware. **For this reason 2-stage upscaling now defaults to off in the GUI** — enable it deliberately, and start well below the maximum until you've confirmed your system handles it.

The binding constraint is VRAM, not RAM — the transformer's attention cost scales with `latent_frames × (H/32) × (W/32)`, which the script tracks as a "token count" and warns you about above ~50,000 (with the option to continue anyway). That warning threshold is a rough estimate, not a guarantee — the 193-frame lockup above was inside it.

| Mode | Base resolution | Frames | Final output | Notes |
|---|---|---|---|---|
| **Maximum recommended** | 960x544 (2-stage) | up to 121 (5s @24fps) | **1920x1088** | The practical ceiling on 16GB VRAM — don't go higher without headroom to spare |
| Single-stage, no upscale | 1280x704 | up to ~121f | 1280x704 | Skip the checkbox for a direct full-res render |
| ~~Pushing it~~ — do not use | ~~960x544 (2-stage), up to 193f~~ | | ~~1920x1088~~ | **Locked up the whole machine during testing. Not recommended at any frame count above the row on this GPU.** |

Going past the maximum recommended row (e.g. full-res 1280x704 direct at 2x, or frame counts above 121 with upscaling on) risks exhausting 16GB of VRAM, tripping the ring-timeout watchdog, or freezing the system outright — the token-count warning dialog is a rough guide, not a safety guarantee, so don't rely on it alone.

> ⚠️ **On 32GB RAM (the minimum), do not attempt the "Pushing it" row or the highest resolution/frame settings.** Those figures assume the 124GB this script was tuned for — resident model caching (~18GB pinned) plus a transient 23GB text-encoder subprocess plus VAE/upsampler staging can exceed 32GB well before VRAM becomes the limit, causing a system-RAM OOM rather than a clean VRAM error. At 32GB, stick to the default 960x544 → 1920x1088 preset at the lower end of the frame range (121f), click **🧹 Free Models** between runs, and consider dropping `blocks_per_group` to 2.

### Tuning knobs (in `ltx2_config.json`)

* `blocks_per_group` — offload group size for the transformer's 48 blocks (default 4 → 12 groups). Raise to 6–8 for more speed at the cost of steady-state VRAM; drop to 2 if the upscale/refinement stage runs out of memory.
* `attention_backend` — defaults to `native` (PyTorch SDPA auto-dispatch, which picks the right kernel for gfx1201 automatically). Only override this if you're deliberately experimenting — a forced kernel can hard-fail on masked cross-attention.

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

### Out of Memory (VRAM) Errors
1. Drop `blocks_per_group` to 2 in `ltx2_config.json`.
2. Reduce resolution or frame count (see table above).
3. Click **🧹 Free Models** before a large run if you've been experimenting with other settings — a stale resident pipeline plus a big new allocation can add up.

System RAM is no longer the limiting factor **on a 124GB machine** (peak usage is roughly 45GB — the resident 18GB transformer plus a transient text-encoder subprocess), so a large swap file there is optional insurance rather than a hard requirement. On a 32GB machine it's the opposite: that same 45GB peak won't fit in RAM alone, so a swap file (16GB+) is required, not optional, and you should still expect to stay off the higher resolution/frame settings — see the warning above.

---

## Requirements

* AMD GPU with ROCm support (developed/tested on RX 9070 XT, gfx1201)
* **System RAM: 32GB minimum.** This is a hard floor, not a comfortable one — see the RAM warning under Recommended Settings above. At 32GB, keep to the lower end of resolution/frame settings and expect to rely on swap; the higher-end presets (large 2-stage output, longer clips) were tuned for and tested on 124GB and are not safe to attempt at 32GB.
* Pre-quantized FP8 weights in `./local_ltx25_fp8` (run `quant_transformer_fp8.py` once if you don't have these)
* The base LTX-2.5 model directory (`./local_ltx25_model`) for the latent upsampler config used by the 2-stage upscale path
