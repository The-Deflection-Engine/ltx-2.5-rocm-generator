<!--
MAINTENANCE: When modifying any pipeline class in src/ltx_pipelines/,
update this document to reflect changes to:
- __init__ / __call__ signatures
- sigma handling or step counts
- denoiser types or guidance
- new or removed pipelines
Run: ls src/ltx_pipelines/*.py to check for new pipeline files.
-->

# ltx-pipelines

Inference pipelines for LTX-2 audio-video generation. Depends on `ltx-core` for model definitions, diffusion components, and loading. All pipelines live in `packages/ltx-pipelines/src/ltx_pipelines/`.

## Pipeline selection

| Pipeline | File | Stages | Model | Sampler | Use case |
|----------|------|--------|-------|---------|----------|
| `TI2VidOneStagePipeline` | `ti2vid_one_stage.py` | 1 | Full | Euler | Simple text/image-to-video |
| `T2AOneStagePipeline` | `t2a_one_stage.py` | 1 | Full | Euler | Text-to-audio (audio-only output, no video branch) |
| `TI2VidTwoStagesPipeline` | `ti2vid_two_stages.py` | 2 | Full + distilled LoRA | Euler | Production quality |
| `TI2VidTwoStagesHQPipeline` | `ti2vid_two_stages_hq.py` | 2 | Full + distilled LoRA (both stages) | Res2s | Highest quality, fewer steps |
| `A2VidPipelineTwoStage` | `a2vid_two_stage.py` | 2 | Full + distilled LoRA | Euler | Audio-conditioned video |
| `KeyframeInterpolationPipeline` | `keyframe_interpolation.py` | 2 | Full + distilled LoRA | Euler | Keyframe interpolation |
| `DFRPipeline` | `dfr_pipeline.py` | 2 (+ optional tiled temporal) | Keyframe-slot SFT + distilled LoRA (+ detailing IC-LoRA stage 2) | Euler | Keyframe slots → spatial detailing → optional tiled temporal x2 |
| `DistilledPipeline` | `distilled.py` | 2 | Distilled only | Euler (stage 1: ancestral on 2.5+) | Fastest inference |
| `ICLoraPipeline` | `ic_lora.py` | 2 | Distilled only | Euler | Video-to-video with IC-LoRA control |
| `DubItPipeline` | `dubit.py` | 2 | Distilled only | Euler | Dub-It with IC-LoRA + audio ref conditioning |
| `RetakePipeline` | `retake.py` | 1 | Full or distilled | Euler | Video region regeneration |

## Guidance

- **CFG**: Blends conditioned/unconditioned predictions. Defaults: `cfg_scale=3.0` (video), `7.0` (audio).
- **STG**: Perturbs self-attention in transformer blocks. Default `stg_scale=1.0`, `stg_blocks=[28]` (LTX-2.3) / `[29]` (LTX-2). HQ disables STG (`stg_scale=0.0`).
- **Modality guidance**: Cross-modal attention scaling (`modality_scale=3.0`).
- All guidance is stage 1 only. Stage 2 always uses `SimpleDenoiser`.

## Sigma schedules and step counts

- **Scheduler-based** (full model): `self._scheduler = LTX2Scheduler()` with `execute(steps=N)` (HQ also passes `latent=` for token-count-dependent shift). Defaults: 30 steps (LTX-2.3), 40 (LTX-2), 15 (HQ).
- **Distilled**: Fixed 8-step `DISTILLED_SIGMA_VALUES` (9 values). Stage 2 uses 3-step `STAGE_2_DISTILLED_SIGMA_VALUES` (4 values). No `num_inference_steps` param.
- **Retake**: `num_inference_steps=40` default; ignored when `distilled=True` (fixed 8-step).
- **Overrides**: All pipelines accept optional sigma tensors in `__call__`: `sigmas` (one-stage), `stage_1_sigmas` + `stage_2_sigmas` (two-stage).

## LoRA conventions

- No default LoRAs. `loras` param defaults to empty list/tuple. `DEFAULT_LORA_STRENGTH = 1.0`.
- Two-stage non-distilled pipelines require `distilled_lora` (applied to stage 2 only in TI2Vid/A2Vid/Keyframe).
- HQ applies distilled LoRA to **both** stages with separate `distilled_lora_strength_stage_1` / `_stage_2` params.
- DFR also applies distilled LoRA to **both** stages (shared stage object for stage 1 /
  temporal rounds; stage 2 may add a detailing IC-LoRA via `with_loras`).

## Shared building blocks (`utils/blocks.py`)

- `DiffusionStage` -- owns transformer lifecycle; builds model on call, frees on exit via `gpu_model()` context manager (moves params to meta device to release GPU/CPU memory). Accepts optional `stepper` and `loop` overrides. `__init__` takes a pre-built transformer builder; pipelines construct it via the `DiffusionStage.from_checkpoint(checkpoint_path, ..., loras=...)` classmethod, which builds the standard (and, when offloading, streaming) builders. `with_builder` / `with_loras` return a new stage with a swapped builder / LoRA set without re-specifying config.
- `PromptEncoder` -- Gemma text encoder + embeddings processor (video 4096-dim, audio 2048-dim).
- `ImageConditioner` / `AudioConditioner` -- temporary encoder scope; builds encoder, passes to callable, frees. `ImageConditioner` is additionally the single owner of the image-conditioning CRF: `resolve_crf(images)` (called near the top of every pipeline's `__call__`) fills in the H.264 CRF of any `ImageConditioningInput` that left it unset, reading it from the checkpoint's `model_version` (`detect_params`, lazily and cached). So omitting `crf` means "use what matches this model" (33 through LTX-2.3, 18 from 2.4), while an explicit `crf` -- including `0` for no re-compression -- is always honoured.
- `VideoUpsampler` -- 2x spatial upsampling via encoder + upsampler.
- `VideoDecoder` / `AudioDecoder` -- latent-to-pixel decoding (iterator for video, `Audio` for audio). `VideoDecoder` takes a single decoder `checkpoint_path` and `diffvae_optimization` (`DiffVAEMode`, default `CHUNKED_EAGER`); pipelines pass `model_paths.video_vae()`, the same file `ImageConditioner` and `VideoUpsampler` build their encoder from -- a video VAE checkpoint carries encoder and decoder together. Decoder kind (conv vs diffusion) is chosen from checkpoint metadata (`is_diffusion_video_vae`); distilled DiffVAE uses fixed 2 Euler steps. CLI: `--video-vae-path` is the video VAE slot in both monolith (optional override; defaults via `ModelPaths.from_monolith`) and split modes; `--diffvae-optimization` (see `docs/optimization.md#diffusion-vae-decoder` for mode meanings and relative compile/runtime/VRAM factors). Multi-GPU pipelines (`*_mgpu.py`) accept the same flags and wrap the loaded decoder in `DistributedVideoDecoder`.
- `DurationPredictor` -- predicts a frame count from `PromptEncoder`'s connector token outputs via `DurationHead` (`ltx_core.duration_head`), snapped to the VAE's temporal grid. Unlike other blocks, it holds the built model directly (`__init__` takes a `DurationHead`, not a builder) since the checkpoint is only a few MB -- no build-on-call/free-on-exit needed. Pipelines construct it via `DurationPredictor.from_checkpoint(checkpoint_path, dtype, device)`, which returns `None` instead of a predictor if the checkpoint has no `duration_head.*` weights (checkpoints predating LTX-2.5 / gemma4). Called with `(video_encoding, audio_encoding, frame_rate=...)` -- either may be `None` but not both. Wired into `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`, `DistilledPipeline`, `DFRPipeline`, and `T2AOneStagePipeline` (audio-only: `video_encoding=None`), plus their MGPU runners: when `num_frames` is omitted (`None`), `require_num_frames_source` (called at the top of `__call__`, before any work) raises immediately if no `DurationPredictor` is available, otherwise it's auto-predicted from the caption. The two-stage pipelines (`TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`, `DistilledPipeline`, `DFRPipeline`) and their MGPU runners return the resolved `num_frames` as a third tuple element from `__call__`, so callers can compute `get_video_chunks_number` for the progress bar without duplicating the auto-duration fallback logic.

### Memory management

- **Model lifecycle**: All blocks build their model on call and free it on exit. `gpu_model()` moves params to `"meta"` device on exit, immediately releasing storage. No model persists between calls.
- **Block streaming**: When offloading is enabled, `DiffusionStage` wraps the transformer in `BlockStreamingWrapper`. Blocks live on pinned CPU memory; only 2 blocks are buffered on GPU at a time (one for compute, one for async H2D copy on a separate CUDA stream).
- **Batch splitting**: `BatchSplitAdapter` wraps the transformer and splits inputs exceeding `max_batch_size` into sequential chunks. If guidance needs B=4 but `max_batch_size=1`, it runs 4 sequential B=1 passes. Higher `max_batch_size` reduces layer-streaming PCIe transfers at the cost of peak memory.

## Denoisers (`utils/denoisers.py`)

- `SimpleDenoiser` -- single forward pass (B=1), no guidance. Used by distilled pipelines and all stage 2.
- `GuidedDenoiser` -- CFG/STG with static `MultiModalGuider` instances (HQ, A2Vid, Retake non-distilled).
- `FactoryGuidedDenoiser` -- per-step guider creation via factory (OneStageTI2Vid, TwoStagesTI2Vid, Keyframe).

All denoisers return a `(video_result, audio_result)` tuple of `DenoisedLatentResult` (defined in `utils/types.py`), either element may be `None` for absent modalities. `DenoisedLatentResult.denoised` is the final blended tensor. Guided denoisers additionally populate per-pass fields (`.cond`, `.uncond`, `.ptb`, `.mod`) on each result; `SimpleDenoiser` leaves these `None`.

`GuidedDenoiser` and `FactoryGuidedDenoiser` accept `force_uncond_pass=True` to run the uncond pass even when `cfg_scale=1.0` (required by CFG++ when the guidance scale is 1 but the uncond prediction is still needed for the ODE derivative). Requires `negative_context` to be set on the guider. When enabled, `DenoisedLatentResult.uncond` will be a tensor instead of `None`.

Guided denoisers batch all guidance passes into a **single transformer call**: states are repeated along the batch dimension, contexts concatenated, and a `BatchedPerturbationConfig` controls which attention ops are skipped per sample. Pass count is dynamic: B=2 for CFG-only, up to B=4 with CFG+STG+modality isolation. Results are split back and blended by the guider.

## Per-pipeline unique features

- **HQ**: Res2s second-order sampler for **both** stages, latent-dependent sigma schedule, distilled LoRA on both stages with separate strengths.
- **A2Vid**: Audio frozen in both stages (`frozen=True, noise_scale=0.0`). That zeros `denoise_mask` and forces audio `Modality.sigma=0` (prompt AdaLN / a2v gate). Returns original audio (not VAE-decoded); no `AudioDecoder`.
- **IC-LoRA**: `VideoConditionByReferenceLatent`, `reference_downscale_factor` from LoRA metadata, `skip_stage_2`, attention mask downsampling. Stage 2 is LoRA-free and uses `combined_image_conditionings` (no IC-LoRA conditioning).
- **Dub-It**: Standalone pipeline; IC reference **video** helpers in `iclora_utils.py`, Dub-It-only **audio** patchify/negative positions in `dubit.py`. Appends frozen audio-reference tokens via `AudioConditionByReferenceLatent` (ltx-core), matching video token order (`[target | ref]`) while keeping reference RoPE positions negative (training-compatible). Single IC-LoRA on both stages; full IC-LoRA video conditioning at stage 1 and 2; stage-2 audio is frozen with S1 latent as initial state and uses S1-derived ref. Final audio decoded from stage 1 latent. The Dub-It CLI does not expose `--conditioning-attention-mask`; use `ic_lora.py` if you need spatial IC attention masking.
- **Keyframe**: Uses `image_conditionings_by_adding_guiding_latent` in both stages (all frames as keyframe guidance, no replacement) -- unlike TI2Vid which uses `combined_image_conditionings` (frame_idx=0 replaces, others guide).
- **Retake**: `TemporalRegionMask` for selective time-window regeneration. `regenerate_video`/`regenerate_audio` flags. Conditional distilled/full behavior.
- **Distilled**: Single `self.stage` reused for both stages (not `stage_1`/`stage_2`). Stage 1's sampler is resolved in `__init__` from the checkpoint generation -- `should_use_ancestral_sampler(path)` (`detect_model_version(...) >= (2, 5)`) sets `self.use_ancestral_sampler`, so LTX-2.5+ uses `EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0)` (the reference sampler's own defaults) + `euler_ancestral_denoising_loop` and older checkpoints keep plain Euler. There is no per-call override; assign the attribute before calling to pin a sampler (e.g. to A/B both on one checkpoint). `distilled_mgpu.py` detects per rank from the same checkpoint, so ranks stay symmetric. Stage 2 is always plain Euler -- its 3-step schedule is too short to remove freshly injected noise. The loop's noise seed is offset (`ANCESTRAL_NOISE_SEED_OFFSET`) so its first draw is not bit-identical to the initial `GaussianNoiser` noise.
- **DFR** (Diffusion Fidelity Rendering, `dfr_pipeline.py` + its `dfr_layout.py` sidecar):
  distilled-schedule pipeline on a keyframe-slot-capable **SFT** base with a **distilled LoRA**
  (strength 1.0). Stage 1 (half-res) generates video + keyframe slots on an **x8-border segment
  grid**: pad ``(num_frames-1)`` up to a multiple of S (prefer S=32 unless S=24 pads strictly less);
  positions ``S, 2S, …, N'-1``. Half-res video is reserved for IC-LoRA; video and slot keyframes are
  spatially latent-upsampled. Stage 2 jointly denoises with distilled LoRA and an optional x2
  detailing IC-LoRA (no default -- pass ``--detailing-lora`` to enable it; ``VideoConditionByReferenceLatent`` on the
  reserved half-res stage-1 video, ``STAGE_2_DISTILLED_SIGMAS``). Shipped audio is **stage 1's**:
  stage 2 still runs an audio pass (video needs the cross-modal attention) but re-noises audio under
  the detailing LoRA, and the temporal rounds pass ``audio=None``, so nothing refines it afterwards.
  Optional ``temporal_upsample_rounds`` (0–2): each round temporally x2-upsamples, partitions into
  ``2**round`` keyframe-seam tiles with a one-segment lead-in on non-first tiles, invents
  mid-segment slots per tile, and densifies with ancestral Euler (η=0.5).
  Three invariants worth knowing before touching the temporal rounds:
  - **Stitch handover is exactly at the shared keyframe.** A tile's local latent 0 is an *image*
    latent (1 pixel frame) and local latent 1 was denoised against it; neither may be spliced into
    the mid-canvas stream. The earlier tile keeps through its trailing seam latent and the later tile
    resumes strictly after the KF (``drop_latent_prefix`` covers lead-in + shared seam). Handing over
    *before* the KF loses the synchronization point and shows as a jump just before the seam.
  - **Conditioning fps is capped at 60** (``_MAX_CONDITIONING_FPS``), independently of playback fps.
    RoPE time is ``pixel_frame / fps``, so a 120 fps time base halves every token's temporal span
    versus the trained distribution and the model can no longer lay out the 8 pixel frames inside a
    latent -- it decodes as a motion spike at each 8-frame latent border followed by a stall. Only
    decode/encode use playback fps.
  - **Image conditioning is tile-local.** ``frame_idx=0`` means the *tile's* first frame, so
    re-applying the opening image on a non-first tile pins the wrong frame onto the seam; only images
    that fall inside the window are re-attached, remapped to local indices.

  The canvas may pad the tail up to a whole segment, but the caller always gets
  ``(requested_frames - 1) * 2**rounds + 1`` frames: the excess is trimmed before decode (always on
  a latent boundary, since ``requested - 1`` is a multiple of the VAE temporal scale), and audio is
  cut to the video's duration. CLI: ``--detailing-lora``, ``--temporal-upsampler-path``,
  ``--temporal-upsample-rounds``.
  No ``--num-generated-keyframes``.

## Image conditioning helpers (`utils/helpers.py`)

- `combined_image_conditionings()` -- images with `frame_idx==0` replace latent (`VideoConditionByLatentIndex`), others guide (`VideoConditionByKeyframeIndex`).
- `image_conditionings_by_adding_guiding_latent()` -- all images become keyframe guidance regardless of `frame_idx`.
- `evenly_spaced_keyframe_positions()` -- evenly spaced **interior** positions (both endpoints excluded).

## Generated keyframes

Optional, off by default on most pipelines, and requires a checkpoint whose transformer config sets
`use_keyframes_abs_pos_embedding`. Appends empty, fully-denoised single-pixel-frame token slots at
interior frame positions so the model generates extra frames there, relaxing the effective temporal
compression at those positions. Slots may optionally carry ``initial_keyframes`` latent seeds
(written into the appended ``latent`` tokens; ``denoise_mask=1`` still applies).

- **Where**: first stage only, on `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`,
  `TI2VidTwoStagesHQPipeline`, `DistilledPipeline`, `DFRPipeline`, and the three
  `*_mgpu` runners. Stage 2 never gets *slots* on TI2Vid/Distilled (effect is baked into the
  stage-1 latent). `DFRPipeline` is the exception: stage 2 re-attaches **seeded** slots
  (spatially upsampled stage-1 KF latents) and optional tiled temporal rounds attach
  mid-segment slots per tile.
- **API**: `generated_keyframes: int | Sequence[int] = 0` on `__call__` for TI2Vid/Distilled
  (an `int` requests evenly spaced **interior** keyframes; a sequence gives explicit indices).
  CLI: `--num-generated-keyframes` via `add_generated_keyframes_arg`. `DFRPipeline` always uses
  the x8-border segment grid from `dfr_layout.resolve_canvas` and does **not** expose that arg;
  it adds `--detailing-lora` (optional, no default),
  `--temporal-upsampler-path`, `--temporal-upsample-rounds`.
- **Validation**: `DiffusionStage.supports_generated_keyframes` reads the checkpoint config, and
  `__call__` raises if slots are requested without it -- each keyframe costs a full latent frame of
  tokens, so silently degrading would waste 16-31% of the token budget.
- **Cost**: one latent frame of tokens per keyframe, yielding 1 pixel frame instead of 8. At
  512x768/241f, 5 keyframes is +16% tokens (~1.35x attention); at 1088x1920/121f it is +31% (~1.72x).
- **Reading the keyframes back**: `LatentState.generated_keyframes` (`(B, C, K, H, W)`), extracted by
  `clear_conditioning` using the `generated_keyframe_layout` recorded on the state -- exact, not
  positional slicing. From outside a pipeline, substitute `RecordingDiffusionStage` for
  `pipeline.stage`. Decode each keyframe as a standalone one-frame clip; a K-frame causal decode
  would blend slots that were never adjacent. See `internal/scripts/generated_keyframe_diagnostics.py`.
- **Invariants**: slots are *appended* (which preserves target noise across a same-seed
  keyframes/no-keyframes A/B at B=1) and pass `attention_mask=None` (a dense `(B, T, T)` mask would
  be ~1.8 GB at 30k tokens and would disable FA3/FA4).
