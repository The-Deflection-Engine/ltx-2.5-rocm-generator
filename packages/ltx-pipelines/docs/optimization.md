# ⚡ Optimization Tips

## Memory Optimization

### FP8 Quantization (Lower Memory Footprint)

For smaller GPU memory footprint, use the `--quantization` flag and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Quantization policies available via `--quantization`:

| Policy | CLI Flag | Description |
| ------ | -------- | ----------- |
| **FP8 Cast** | `--quantization fp8-cast` | Downcasts transformer linear weights to FP8 during loading; upcasts on the fly during inference. No extra dependencies. |
| **FP8 Scaled MM** | `--quantization fp8-scaled-mm` | Uses FP8 scaled matrix multiplication via PyTorch's `torch._scaled_mm`. Best performance on Hopper+ GPUs with native FP8 support. |
| **NVFP4 Cast** | `--quantization nvfp4-cast` | Online BF16→NVFP4 for allowlisted DiT Linears via `ltx_kernels.nvfp4`. Requires Blackwell (SM ≥ 10) and `nvfp4_cpp` (`uv sync --group kernels` / `TORCH_CUDA_ARCH_LIST=10.0`). |
| **NVFP4 prequant** | `--quantization nvfp4-prequant` | Load a pre-quantized NVFP4 checkpoint (packed uint8 weights + block scales + weight `weight_scale_2` + calibrated `input_scale`). Uses static activation per-tensor scale from the checkpoint (no runtime amax); per-16 E4M3 block scales are still computed each forward. Pair with a BF16 VAE. Same SM / `ltx-kernels` requirements as cast. Every discovered NVFP4 Linear is hosted as `NVFP4Linear`, including layers whose `weight_scale` rows are cuBLAS-padded above `out_features` (no BF16 dequant fallback). |

**CLI:**

```bash
# FP8 Cast (works on any GPU with FP8 support)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.ti2vid_two_stages \
    --quantization fp8-cast --checkpoint-path=...

# FP8 Scaled MM (no extra deps, best on Hopper+ GPUs)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.ti2vid_two_stages \
    --quantization fp8-scaled-mm --checkpoint-path=...

# NVFP4 from a pre-quantized DiT checkpoint (BF16 VAE)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.distilled \
    --quantization nvfp4-prequant \
    --distilled-checkpoint-path=/path/to/nvfp4.safetensors \
    --vae-checkpoint-path=/path/to/bf16.safetensors
```

**Programmatically:**

When authoring custom scripts, pass a `QuantizationPolicy` to pipeline classes:

```python
from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
# Alternatives:
# from ltx_core.quantization.fp8_scaled_mm import build_policy as build_fp8_scaled_mm_policy
# from ltx_core.quantization.nvfp4 import build_nvfp4_cast_policy, build_nvfp4_prequant_policy

pipeline = TI2VidTwoStagesPipeline(
    model_paths=ModelPaths.from_monolith(ltx_model_path, gemma_root_path),
    distilled_lora=distilled_lora,
    spatial_upsampler_path=upsampler_path,
    loras=[],
    quantization=build_fp8_cast_policy(ltx_model_path),
)
pipeline(...)
```

You still need to use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when launching:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python my_denoising_pipeline.py
```

### Memory Cleanup Between Stages

By default, pipelines clean GPU memory (especially transformer weights) between stages. If you have enough memory, you can skip this cleanup to reduce running time:

```python
# In pipeline implementations, memory cleanup happens automatically
# between stages. For custom pipelines, you can skip:
# utils.cleanup_memory()  # Comment out if you have enough VRAM
```

## Compilation (`torch.compile`)

Compiling the transformer blocks with `torch.compile` speeds up inference. It is **opt-in and off by default**. The blocks are compiled shape-polymorphically (the sequence dimension is marked dynamic by default; see `seq_dim_dynamic`), so one compiled artifact serves any token count without recompiling.

**CLI** - the `--compile` flag maps directly to `CompilationConfig`:

| Form | Result |
| ---- | ------ |
| *(flag absent)* | eager, no compilation |
| `--compile` | compile with defaults |
| `--compile KEY=VALUE ...` | compile, overriding individual fields |

```bash
# Defaults
python -m ltx_pipelines.ti2vid_two_stages --compile --checkpoint-path=...

# reduce-overhead / max-autotune / capture all need GPU-resident weights
# (keeps_gpu_resident_weights -- SP/TDP + weight tracker, or streaming).
python -m ltx_pipelines.ti2vid_two_stages --compile mode=reduce-overhead --checkpoint-path=...

# Several overrides at once
python -m ltx_pipelines.ti2vid_two_stages \
    --compile mode=max-autotune fullgraph=true dynamic=true --checkpoint-path=...

# Self-managed block-loop CUDA-graph capture
python -m ltx_pipelines.ti2vid_two_stages --compile capture=true --checkpoint-path=...
```

| Field | Values | Default | Notes |
| ----- | ------ | ------- | ----- |
| `mode` | `none`, `reduce-overhead`, `max-autotune`, … | `none` | `reduce-overhead`/`max-autotune` enable CUDA graphs; require `keeps_gpu_resident_weights=True` or the stage raises |
| `backend` | `inductor`, `eager`, … | `inductor` | |
| `fullgraph` | `true`/`false` | `false` | |
| `dynamic` | `auto`/`true`/`false` | `auto` | the seq dim is marked dynamic separately (see `seq_dim_dynamic`) |
| `inductor_config` | JSON object or path to a `.json` | `{}` | `torch._inductor.config` overrides |
| `dynamo_config` | JSON object or path to a `.json` | `{"inline_inbuilt_nn_modules": true, "cache_size_limit": 256}` | `torch._dynamo.config` overrides |
| `seq_dim_dynamic` | `true`/`false` | `true` | mark the block's sequence dim dynamic so one artifact serves any token count; `false` recompiles per token count |
| `recompile_perturbed_block` | `true`/`false` | `true` | `true` recompiles a separate block graph for the STG-perturbed pass; `false` attaches unconditional runtime masks so the block traces once (single graph) |
| `capture` | `true`/`false` | `false` | per-block compile + one CUDA graph over the block loop (keyed by shape / perturbation signature). Requires `keeps_gpu_resident_weights=True`. Faster cold-start and smaller new-shape penalty than `mode=reduce-overhead` |

**Controlling inductor / dynamo configs.** `inductor_config` and `dynamo_config` take either an inline JSON object or a path to a `.json` file, applied via `torch._inductor.config.patch(...)` / `torch._dynamo.config.patch(...)` around the compiled forward. They **replace the defaults wholesale - they do not merge**, so when overriding `dynamo_config` re-include any defaults you want to keep:

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --compile 'inductor_config={"max_autotune": true}' \
              'dynamo_config={"inline_inbuilt_nn_modules": true, "cache_size_limit": 256, "recompile_limit": 32}' \
    --checkpoint-path=...
```

**Programmatically**, pass a `CompilationConfig` to the pipeline:

```python
from ltx_core.model.transformer.compiling import CompilationConfig

pipeline = TI2VidTwoStagesPipeline(
    ...,
    compilation_config=CompilationConfig(mode="reduce-overhead"),
)
```

**Faster cache loads: `unsafe_skip_cache_dynamic_shape_guards` (unsafe, opt-in).** Inductor's FX-graph cache re-checks the dynamic-shape guards stored with each entry on every lookup. Setting this flag skips that re-check (every entry is treated as a guard hit), which speeds up warm and cross-process cache loads. It is **not enabled by default** because it is a correctness hazard: a kernel first compiled at a small sequence length keeps int32 address arithmetic, and reusing it at a larger sequence length (roughly **>58k tokens/rank**) overflows int32 and reads out of bounds - surfacing as a CUDA illegal memory access or silently corrupted output. Only enable it when your token counts stay within the range the cached kernels were compiled for:

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --compile 'inductor_config={"unsafe_skip_cache_dynamic_shape_guards": true}' \
    --checkpoint-path=...
```

## Diffusion VAE Decoder

By default, the video VAE comes from `--checkpoint-path`/`--distilled-checkpoint-path` (monolith) or `--video-vae-path` (split). You can point `--video-vae-path` at a **separate** `.safetensors` (e.g. a distilled DiffVAE) without changing the main transformer checkpoint. Decoder kind (conv `ConvVideoDecoder` vs NATTEN `DiffusionVideoDecoder`) is selected from that file's metadata, not from whether the override flag was passed.

> The path supplies the **whole** video VAE, not just the decoder: image conditioning (`ImageConditioner`) and the stage-2 `VideoUpsampler` build their encoder from the same file. Released VAE checkpoints bundle encoder and decoder together.

**Neighborhood attention backends** (selected at DiffVAEMode apply-time):

| Backend | When | Notes |
| ------- | ---- | ----- |
| **CuTe DSL** (`blackwell_dsl`) | Datacenter Blackwell (B200) with `ltx-kernels` / `nvidia-cutlass-dsl` | Fastest on B200; fused stage-5 context projection. Not used on consumer Blackwell (sm_120). |
| **NATTEN** | `natten` extra installed | Fastest production path on non-B200 GPUs (Hopper/Ampere/Ada). |
| **Triton** `na3d` | CUDA + working Triton (incl. [triton-windows](https://github.com/triton-lang/triton-windows)); natten missing | Compatibility fallback. |
| **eager** tiled SDPA | Always | Slowest; last resort when Triton is unavailable. |

```bash
# Recommended for DiffVAE production decode on non-B200 GPUs
uv sync --package ltx-core --extra natten

# B200 / datacenter Blackwell CuTe DSL path
uv sync --group kernels
```

That `natten` extra pins `natten==0.21.7+torch2130cu132` with `torch==2.13.0` (cu132). If DiffVAE decode hits a CUDA illegal memory access, upgrade CUDA / PyTorch / natten to those pins - the failure usually comes from outdated NVIDIA libs bundled with older PyTorch, not from tiling config.

**Mode × backend rules:**

- `blackwell_dsl` uses CuTe DSL NA + fused stage-5 (requires datacenter Blackwell + `uv sync --group kernels`). Independent of natten.
- `combined_compile` **requires natten** (full-volume NA + `torch.compile`). Without it, apply raises.
- `chunked_eager` / `chunked_compile` remapped to Triton/eager when natten is missing (compile is disabled on the fallback path). Triton/eager are for compatibility only.

Pipeline `AUTO_TILING` resolves via `tiling_config_for_vae`: for a **Conv VAE** that is aspect-coupled long-side `768/64` spatial tiles plus temporal `80/24` (`TileSizeConfig.from_long_side`). For **DiffVAE**, recommended decode tiling enumerates legal per-axis tile **sizes** on the VAE grid (via the same `split_by_size` path decode uses) and emits a `TileSizeConfig` with independent `frames` / `height` / `width` `DimensionSizeConfig` values. Tile sizes are floored at ``2xOverlap`` so blend masks stay complementary. Stage-5 VRAM coef for `chunked_compile` is **7** with natten; with Triton/eager fallback it uses the `chunked_eager` coef (**5**).

**CLI:**

| Flag | Effect |
| ---- | ------ |
| `--video-vae-path <path>` | Video VAE `.safetensors` path (encoder + decoder). Split: the `vae/` component. Monolith: optional override; defaults to `--checkpoint-path` / `--distilled-checkpoint-path` when omitted. |
| `--diffvae-optimization` | DiffVAE preset: `chunked_eager` (default), `chunked_compile`, `combined_compile`, `blackwell_dsl`. Ignored for convolutional VAEs. |

**Modes** (default `chunked_eager`):

| Mode | Behavior |
| ---- | -------- |
| `chunked_eager` | Deferred stage-4 inject (sequential upsample then `context_proj`), W-chunks=4, cutlass-fna (or Triton/eager fallback), no `torch.compile`. Lowest compile cost; lower peak VRAM. |
| `chunked_compile` | Same deferred/chunked pathway + `torch.compile` on attn+mlp only when natten is present (det stages stay eager). Without natten: same as `chunked_eager` fallback path. |
| `combined_compile` | Combined `context_and_x` buffer, full-volume attention, compile blocks + det stages. **Requires natten.** Fastest warm decode on non-B200; highest VRAM. |
| `blackwell_dsl` | Deferred stage-4 + CuTe DSL NA / fused stage-5. Fastest on datacenter Blackwell (B200). Needs `uv sync --group kernels`. |

**Relative performance** (order-of-magnitude; hardware varies - no absolute timings or VRAM figures):

- **Compile:** `chunked_compile` roughly **~2x faster to compile** than `combined_compile` (det stages not compiled).
- **Warm runtime (non-B200):** relative to `combined_compile` (fastest with natten): `chunked_compile` roughly **~1.4x slower**, `chunked_eager` roughly **~2-2.5x slower**.
- **Warm runtime (B200):** `blackwell_dsl` is the preferred / fastest path; NATTEN modes remain available but are not the B200 default.
- **Peak VRAM:** `chunked_*` roughly **~½** of `combined_compile`.

Checkpoint notes:

- **Bundled DiffVAE** (full model with `vae.decoder.*` fine path): `coarse_*` preview-head weights are ignored at load; decode is **1-step from noise** via `DiffusionVideoDecoder.forward` (no Euler loop) when `default_num_inference_steps == 1`.
- **Standalone distilled DiffVAE**: legacy `gate_msa` / `gate_mlp` (and optional `gate_ctx`) are folded into `attn.proj` / `mlp.w_down` / `context_proj` on load, then dropped. Default schedule is **2 Euler steps**.

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path path/to/checkpoint.safetensors \
    --video-vae-path path/to/diffusion_vae.safetensors \
    --diffvae-optimization combined_compile \
    ...
```

**Programmatically**, pipelines take a single `model_paths: ModelPaths`; its `video_vae_path` feeds the `VideoDecoder`, `ImageConditioner` and `VideoUpsampler` blocks. `ModelPaths.from_monolith` accepts the same optional override the CLI exposes:

```python
from ltx_core.model.video_vae.transformer import DiffVAEMode

pipeline = TI2VidTwoStagesPipeline(
    ...,
    model_paths=ModelPaths.from_monolith(
        checkpoint_path,
        gemma_root,
        video_vae_path="path/to/diffusion_vae.safetensors",
    ),
    diffvae_optimization=DiffVAEMode.COMBINED_COMPILE,
)
```

Also supported on the multi-GPU pipelines (`*_mgpu.py`): the same flags flow into the underlying pipeline, and `DistributedVideoDecoder` wraps the loaded decoder when applicable. DiffVAE ModuleOps stay on the inner builder so they apply before the distributed wrap.

### Tile size vs compile latency (int32 address threshold)

Inductor's default address arithmetic stays in **int32** while the decoded tile volume stays under ~2.1e9 elements x channels. Crossing that threshold forces int64 addressing and roughly **~2.5x colder compile**:

```
Threshold: T x (H_px/4) x (W_px/4) x 512 ≤ 2,147,483,647
  ⇔ T x (H_px/4) x (W_px/4) ≤ 4,194,303
```

Examples (latent shape after /4 spatial downsample; product is T·H·W·512):

| Tile / clip | Latent (T,H,W) | T·H·W·512 | int32? | Cold compile (approx.) |
| ----------- | -------------- | --------- | ------ | ---------------------- |
| 512x512x48f tile | (48,128,128) | 0.40e9 | ≤int32 | ~30-35s |
| 768x768x80f tile | (80,192,192) | 1.51e9 | ≤int32 | - |
| 736x1024x120f (1024px on 736x1280) | (120,184,256) | 2.89e9 | OVER (int64) | ~70s+ |
| 1024x1024x120f square | (120,256,256) | 4.03e9 | OVER | - |
| 544x960x121 full non-tiled | (121,136,240) | 2.02e9 | ≤int32 (barely) | - |
| 736x1280x241 full non-tiled | (241,184,320) | 7.27e9 | OVER | - |

**Tradeoff:** bigger tiles → fewer tiles / less overlap waste, but crossing int32 costs ~2.5x cold compile. Prefer **≤768@80f** or **512@48f** when compile latency matters; use **1024@120f** when decode throughput wins.

## Denoising Loop Optimization

**Gradient Estimation Denoising Loop:**

Instead of the standard Euler denoising loop, you can use gradient estimation for fewer steps (~20-30 instead of 40):

```python
from ltx_pipelines.utils import gradient_estimating_euler_denoising_loop

# Use gradient estimation denoising loop
def denoising_loop(sigmas, video_state, audio_state, stepper):
    return gradient_estimating_euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=audio_state,
        stepper=stepper,
        transformer=transformer,
        denoiser=denoiser,
        ge_gamma=2.0,  # Gradient estimation coefficient
    )
```

This allows you to use **20-30 steps instead of 40** while maintaining quality. The gradient estimation function is defined in [`samplers.py`](../src/ltx_pipelines/utils/samplers.py).
