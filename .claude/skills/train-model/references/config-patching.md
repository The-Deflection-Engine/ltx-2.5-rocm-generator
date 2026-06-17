# Config Patching

How to safely produce `<workspace>/<run-name>/config.yaml` from an example in `packages/ltx-trainer/configs/`. The trainer's config schema is Pydantic with `extra="forbid"` — unknown fields are rejected. Full field reference: [`packages/ltx-trainer/docs/configuration-reference.md`](../../../../packages/ltx-trainer/docs/configuration-reference.md).

## Workflow

1. Copy the example config matching the selected mode (see `mode-selector.md`) to `<workspace>/<run-name>/config.yaml`.
2. Patch fields as described below. Preserve YAML comments where possible — they help the user audit the run later.
3. **Never** edit the example config in `packages/ltx-trainer/configs/`. That's the user's reference library.

## Required Patches (every run)

| Field | Value |
|-------|-------|
| `model.model_path` | Absolute path to local `.safetensors` (from probe or user). |
| `model.text_encoder_path` | Absolute path to local Gemma directory (from probe or user). |
| `data.preprocessed_data_root` | `<workspace>/<run-name>/dataset/.precomputed` (absolute). |
| `output_dir` | `<workspace>/<run-name>/outputs` (absolute). |

## Hardware-Driven Patches

Apply per the matched VRAM tier in `references/hardware-profiles.md`. After autotune (Phase 6), patch the winning trial's deltas in.

## Schema Constraints (validate before launch)

These will cause Pydantic errors or runtime failures; check before invoking the trainer.

- **Frame count:** `validation.video_dims[2]` must satisfy `frames % 8 == 1` (1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, ...).
- **Resolution:** `validation.video_dims[0]` and `[1]` must be divisible by 32.
- **Multi-bucket training:** if dataset uses multiple resolution buckets, set `optimization.batch_size: 1`.
- **At least one generated modality:** `training_strategy` must have at least one of `video.is_generated` or `audio.is_generated` set to `true`.
- **Audio condition restrictions:** the audio modality cannot use `first_frame` or `spatial_crop` conditions.
- **Strategy name:** prefer `training_strategy.name: "flexible"`. `text_to_video` and `video_to_video` still work but emit deprecation warnings.

## LoRA Patches

For style/concept LoRAs:

- `lora.rank` and `lora.alpha`: set from the matched VRAM tier and use case. **32GB tier** pins rank 16 per `t2v_lora_low_vram.yaml`; **80GB+ tier** uses rank 32 per `t2v_lora.yaml`. Keep `alpha == rank`. See `mode-selector.md` for use-case-driven rank guidance.
- `lora.target_modules`: short patterns like `"to_k"`, `"to_q"`, `"to_v"`, `"to_out.0"` match all attention modules (video + audio + cross-modal). Add `"ff.net.0.proj"`, `"ff.net.2"` only if user explicitly wants higher capacity.
- **Audio-only LoRA targets** (T2A, audio inpainting): use `"audio_attn1.to_*"`, `"audio_attn2.to_*"` patterns to avoid touching video weights. See `configs/t2a_lora.yaml` for the exact list.

## Validation Sample Prompts

The example configs ship with placeholder validation prompts. Validation condition fields are documented in
[`configuration-reference.md#validation-condition-types`](../../../../packages/ltx-trainer/docs/configuration-reference.md#validation-condition-types).
For style/concept LoRAs:
- Replace at least one `validation.samples[].prompt` with a prompt that uses the user's trigger word or describes the target concept. Tells the user something useful at the first validation interval.
- Keep `validation.video_dims` consistent with the training resolution to make samples comparable.
- **Describe the audio, for any run with a generated audio modality** (joint audio+video, T2A, V2A, etc.). The validation prompts must describe the audio the **same way the training captions do** — if the training captions transcribe speech or characterise sound (e.g. *"he says: ‘…’"*, *"calm spoken voice, quiet room tone"*, *"upbeat acoustic guitar"*), the validation prompts must include comparable audio direction. A prompt with no audio description gives the model no guidance for the audio branch and the generated audio comes out poor. This is not speech-specific — any audio (music, ambience, foley) needs describing. Mirror the structure/level of audio detail found in the dataset captions (inspect a few before writing the prompts).

## W&B Patches

- If the W&B credential check passes (`uv run python -c "import wandb; print(bool(wandb.Api().api_key))"` → `True`): `wandb.enabled: true`, `wandb.project` = `ltx2-<mode>`, `wandb.tags` includes the mode. (Do not use `wandb status` — it falsely reports `api_key: null` when logged in via netrc.)
- If `False`: `wandb.enabled: false`. Surface in plan: *"Not logged in to W&B — run `wandb login` before training to enable tracking."* If the check errored/was ambiguous, ask the user rather than assuming off.

## Output Dir Behaviour

The trainer resumes optimizer/scheduler/step state **only when `model.load_checkpoint` is set** to a checkpoint file; it then looks for a matching `training_state_step_*.pt` next to that file. It does **not** auto-detect prior checkpoints in `output_dir/checkpoints/`. To resume, patch `model.load_checkpoint` to the latest checkpoint. To load weights but skip state restore: `checkpoints.no_resume: true`.

For the orchestrator's resume flow: when resuming an interrupted run, patch `model.load_checkpoint` to the latest checkpoint under `output_dir/checkpoints/`. Leaving it unset starts a fresh run from step 0 even if checkpoints exist on disk.

## Self-Check Before Launch

Before any `python scripts/train.py` invocation:

1. All `model_path`, `text_encoder_path`, `preprocessed_data_root` exist on disk.
2. Frame and resolution constraints satisfied (see above).
3. Generated modalities have matching latents directories under `.precomputed/`.
4. For modes with `reference` condition: `reference_latents/` (and/or `reference_audio_latents/`) exists.
5. For modes with `mask` condition: `video_masks/` (and/or `audio_masks/`) exists.

A failed check at this point is much cheaper than a failed training start.
