# Troubleshooting

Quick lookup for failures during sanity check, preprocessing, or training. For deeper coverage see `packages/ltx-trainer/docs/troubleshooting.md`.

## OOM During Training Step

Order of operations (cheapest first):

1. `optimization.enable_gradient_checkpointing: true` (if not already on).
2. `optimization.batch_size: 1` and increase `gradient_accumulation_steps` to preserve effective batch.
3. `optimization.optimizer_type: "adamw8bit"`.
4. `acceleration.quantization: "int8-quanto"`.
5. Reduce `lora.rank` (32 → 16 → 8). Alpha follows rank.
6. Reduce target resolution (`validation.video_dims` and re-preprocess the dataset at the new resolution).

The last option is expensive — flag it clearly to the user before re-preprocessing.

## OOM During Validation Sample Generation

The validation pass loads decoders + runs CFG/STG inference; it can OOM even when the training step fits.

1. `acceleration.load_text_encoder_in_8bit: true` (trainer config, not the dataset script).
2. `acceleration.offload_optimizer_during_validation: true` (especially for full FT or high-rank LoRA).
3. Reduce `validation.video_dims` (smaller validation than training is fine — it's only for visual feedback).
4. Reduce `validation.inference_steps` (e.g. 30 → 20).
5. Increase `validation.interval` to validate less often.

## NaN Loss

1. Check `acceleration.mixed_precision_mode`: prefer `"bf16"`. If `"fp16"`, switch.
2. Verify dataset latents are well-formed: `uv run python scripts/decode_latents.py <latents-dir> <output-dir> --model-path <model>` (it decodes a whole latents directory, not a single `.pt`) should reconstruct sensibly.
3. Lower `optimization.learning_rate` by 5x.
4. Add `optimization.max_grad_norm: 1.0` (default; verify it's set).
5. If using `quantization`, try `null` — INT8/INT4 quantization can interact badly with poorly-conditioned LoRA inits at high LR.

## Validation Samples Look Wrong but Loss Is Fine

Often not a bug — validation uses simplified inference. For real quality assessment, run a checkpoint through `packages/ltx-pipelines/` after training.

## Trainer Won't Start: Config Validation Error

Pydantic `extra="forbid"` means typos in field names fail loudly. Read the error carefully — it names the offending field and path. Fix and re-launch.

Common offenders:
- `latents_dir` typo or wrong relative path.
- A required field genuinely missing after copying an example (most fields have defaults; check the error message for the exact field path).
- `target_modules` listed at wrong nesting level (must be under `lora:`).

## Trainer Won't Start: Missing Files

- `model_path` not found → re-probe `/models/`, `~/models/`, `$LTX_MODELS_DIR`, or ask the user.
- `text_encoder_path` directory missing the Gemma config → ensure the path is to the Gemma model dir, not its parent.
- `preprocessed_data_root` doesn't contain expected subdirs → re-verify Phase 7 ran for the chosen mode (see `phases/preprocess-dataset.md`).

## Resume Stops Working

The trainer **does not auto-resume from `output_dir`**. Resume happens only when `model.load_checkpoint` is explicitly set to a checkpoint file; the trainer then loads those weights and looks for a `training_state_step_*.pt` next to that file to restore optimizer/scheduler/step. Common pitfalls:
- `model.load_checkpoint` not set or set to the wrong path → fresh run from step 0 even when `outputs/checkpoints/` is full of artifacts. Patch `model.load_checkpoint` to the latest checkpoint.
- `checkpoints.no_resume: true` is set → weights load but state is discarded. Remove the flag if you want a proper resume.
- `training_state_step_*.pt` missing from next to the loaded checkpoint → weights load but step counter resets. Make sure the state file accompanies the checkpoint.
- `training_state_step_*.pt` corrupted (size 0, fails `torch.load`) → trainer falls back to step 0 with a warning.

## Autotune Trial Failed Mid-Sweep

- If trial 2 (quantization off) OOMs: revert to trial 1 and stop the sweep. The 32GB tier baseline is correctly aggressive.
- If trial 3 (adamw) OOMs: revert to adamw8bit. Continue with trial 4 if VRAM headroom allows.
- If trial 4 (batch_size up) OOMs: revert and stop. We've found the ceiling.

Never carry over a failing trial's deltas. Always revert to the last-known-good before the next change.

## Captioning Is Slow

- Local `qwen_omni` is a 30B model served by `serve_captioner.py` (vLLM). If the server won't start or OOMs on launch: keep the default `--quantization fp8` (don't use `bf16` unless ≥66 GiB free VRAM), lower `--gpu-memory-utilization`, or reduce `--max-model-len`. If the GPU can't host a 30B model at all, switch to `gemini_flash`.
- `caption_videos.py --captioner-type qwen_omni` errors connecting → the vLLM server isn't running or `--vllm-url` doesn't match. Start `serve_captioner.py` first and confirm the port.
- For most hardware and for larger datasets, prefer `--captioner-type gemini_flash --num-workers <N>` (needs Gemini auth: `GEMINI_API_KEY`/`GOOGLE_API_KEY` or gcloud/Vertex) — runs anywhere and parallelises; local Qwen needs a heavy GPU and a running server.

## Process_Dataset Errors

- "frames divisible by..." → the video doesn't have enough frames at the requested temporal resolution. Either shorten the requested frame count or use `split_scenes.py` to break long videos.
- "shape mismatch" on existing `.precomputed/` → user requested a different resolution than the existing data. Per the invariants, **stop and ask** — do not overwrite. Offer: reuse at old resolution / re-preprocess to a new dir / abort.

## When To Give Up and Ask The User

If a fix isn't obvious from this file or `packages/ltx-trainer/docs/troubleshooting.md` within two attempts, stop and surface the full error + the steps already tried to the user. Don't loop indefinitely on autonomous fixes — the user has context the agent doesn't (which checkpoints are precious, what they care about preserving, etc.).
