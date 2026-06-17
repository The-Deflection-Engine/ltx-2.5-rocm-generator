# Phase 9 — Post-Train Validate

Procedure document for the `train-model` orchestrator (Phase 9). Read this file in full before acting on post-train validation.

Goal: render the final checkpoint against three prompt categories so the user can inspect the result and form their own judgement. Save outputs in an organized layout. **Do not** prompt for pass/fail verdicts, do not infer causes for failures, do not suggest fixes — soft training failures don't have a clean if/then rule book, and pretending otherwise wastes the user's time.

The orchestrator's hard invariants apply (see `../SKILL.md`).

## What this phase does

1. Collect prompts for three categories.
2. Render the final LoRA against all collected prompts.
3. Save outputs under `<workspace>/<run-name>/outputs/eval/<category>/`.
4. Print the paths and exit.

## Categories

### 1 — In-distribution

A few captions from the training set itself. Tests whether the model learned what it was shown.

- Default: 3 random captions from the dataset metadata (seed 42 for reproducibility).
- Source: `<workspace>/<run-name>/dataset/dataset.json` (the captions used for training, after the held-out split).

### 2 — Out-of-distribution

Prompts the model has never seen, but in the same domain. Tests whether the model generalizes the concept beyond memorized phrasings.

Ask the user once:

> "For out-of-distribution validation, paste 2–3 prompts you'd realistically want to generate at inference time. (If you don't have any specific ones in mind, reply 'default' — I'll use a few generic prompts that include the trigger word.)"

- On `default`: synthesize 3 short prompts using the LoRA's trigger word and a generic scene context (e.g., "<trigger> walking in a forest at dawn"). Note these are generic — they're better than nothing, but real user-style prompts make a stronger test.
- Otherwise: use the user's prompts verbatim.
- **For a run with a generated audio modality, the synthesized prompts must describe the audio** — matching how the training captions describe it (inspect a few from `dataset.json` first). A prompt with no audio direction leaves the audio branch unguided and the generated audio comes out poor. E.g. for the talking-head case include spoken-voice/room-tone direction; for music/ambience/foley describe the sound character. (Categories 1 and 3 reuse the real captions verbatim, so they already carry audio description — this only applies to the synthesized Category 2 prompts.) If the user pasted their own prompts and it's an audio run, and they omitted audio direction, note that the audio may be weak without it.

### 3 — Held-out

Captions from samples that were never seen during training. Tests true generalization, not memorization.

- Source: `<workspace>/<run-name>/dataset/holdout.jsonl` (written by `prepare-dataset` Step 5).
- Use all entries if there are ≤5; otherwise sample 5 with seed 42.
- **If holdout doesn't exist** (dataset was too small, or user-skipped during prepare): print a clear note in the output summary — *"Held-out evaluation skipped: no holdout set was reserved for this run. The post-train eval only covers Categories 1 and 2."* Don't synthesize substitutes.

## Rendering mechanism

Use the trainer's existing validation infrastructure rather than wiring up `ltx-pipelines` from scratch. Create a temporary "validate-only" config and run the trainer with it.

### Step 1 — Build eval config

Copy `<workspace>/<run-name>/config.yaml` to `<workspace>/<run-name>/eval-config.yaml`. Patch:

```yaml
model:
  load_checkpoint: "<absolute path to outputs/checkpoints/<final-lora>.safetensors>"

optimization:
  steps: 1                     # we don't want to train; we want validation to fire
  # Keep batch_size/grad_accum at the run's autotuned values to match its VRAM footprint.

validation:
  skip_initial_validation: false
  interval: 1                  # run validation at step 0 (and at the only training step)
  samples:
    # Inject all collected prompts here, tagged by category in the prompt itself
    # so the output filenames make the category obvious.
    - prompt: "[CAT1-IND] <caption from dataset.json>"
    # ... repeat for each prompt in all three categories
  # Keep video_dims, frame_rate, guidance/STG settings as the trained config.
  # Keep generate_audio consistent with the trained modality config.

checkpoints:
  interval: null               # do not save more checkpoints
  no_resume: true              # load the LoRA's weights but do not restore optimizer/scheduler/step state

output_dir: "<workspace>/<run-name>/outputs/eval"
```

The `[CAT1-IND]`, `[CAT2-OOD]`, `[CAT3-HELDOUT]` tags in the prompt strings make the output MP4 filenames self-describing in the trainer's validation sample directory.

**Attach the mode's conditions to each sample.** A bare `prompt` only validates a pure text-to-X mode (T2V, T2A). For any conditioned mode, the trained model expects the same conditioning at validation time — a prompt with no conditions tests a different task than what was trained, and conditioned modes may fail outright. Add the `conditions` list that matches the run's mode (the trained `config.yaml` `training_strategy` and the example config for the mode are the reference):

| Mode | Add to each sample |
|------|--------------------|
| I2V | `conditions: [{type: first_frame, image_or_video: <frame/clip path>}]` |
| Video extension / suffix | `conditions: [{type: prefix|suffix, ...}]` |
| V2V / AV2AV IC-LoRA | `conditions: [{type: reference, ...}]` (point at a held-out reference) |
| V2A (foley) | `conditions: [{type: video_to_audio, ...}]` |
| A2V | `conditions: [{type: audio_to_video, ...}]` |
| Inpainting (video/audio) | `conditions: [{type: mask, ...}]` |
| Outpainting | `conditions: [{type: spatial_crop, ...}]` |
| A2A IC-LoRA | `conditions: [{type: reference, ...}]` (held-out reference audio) |
| T2V, T2A | none — a bare `prompt` is correct |

Mirror the condition shapes used in the mode's example config under `packages/ltx-trainer/configs/`. For held-out (Category 3) and OOD (Category 2) samples on conditioned modes, draw the conditioning media from the held-out set so the eval stays out-of-distribution.

### Step 2 — Run the trainer in validate-only mode

```bash
cd packages/ltx-trainer
uv run python scripts/train.py "<workspace>/<run-name>/eval-config.yaml"
```

The trainer will load the LoRA, run initial validation against all the prompts, do one trivial training step (which we discard), and exit. Validation samples land in `<workspace>/<run-name>/outputs/eval/samples/`.

### Step 3 — Organize outputs and clean up trainer cruft

The validate-only run is a trainer run, so it inevitably writes throwaway artifacts: an indexed `samples/` dir, a forced final checkpoint (the trainer **always** saves one at the end, regardless of `checkpoints.interval`), and a `training_config.yaml`. Don't leave these around or duplicate the renders.

1. **Move** (don't copy) each generated MP4 from the trainer's indexed `samples/` dir into the category layout, naming by category + a slug of the prompt. Use the index→category mapping you built when constructing `validation.samples`.
2. **Delete the trainer cruft** from the eval dir once the renders are moved: the indexed `samples/` dir, the forced `checkpoints/` dir, and `training_config.yaml`. (These are byproducts of the validate-only hack — there's no config flag to suppress the final-checkpoint save, so clean it up here.)
3. Write **one** manifest, `outputs/eval/prompts.json` (filename → full prompt + category). Don't leave a second copy elsewhere.
4. Remove the temporary `eval-config.yaml` (or keep it under the run's scratch, not in `outputs/`).

Final eval layout — exactly this, nothing else:

```
<workspace>/<run-name>/outputs/eval/
  in-distribution/    <NN>_<prompt-slug>.mp4 ...
  out-of-distribution/ <NN>_<prompt-slug>.mp4 ...
  held-out/           <NN>_<prompt-slug>.mp4 ...   # only if a holdout set existed
  prompts.json        # filename -> full prompt + category (single manifest)
```

No `eval/samples/`, no `eval/checkpoints/`, no `eval/training_config.yaml`, no duplicate manifest.

### Step 4 — Surface paths

Print a tight block, no judgement, no follow-up question:

```
Post-train evaluation complete.

In-distribution renders (<K> samples):
  <workspace>/<run-name>/outputs/eval/in-distribution/

Out-of-distribution renders (<M> samples):
  <workspace>/<run-name>/outputs/eval/out-of-distribution/

Held-out renders (<N> samples):
  <workspace>/<run-name>/outputs/eval/held-out/    # or: "(skipped — no holdout set)"

Open the MP4s and decide for yourself whether the model is good. Soft
training quality is judged by watching the videos, not by a checklist —
there's no substitute for your own eyes here.
```

Return control to the orchestrator. The orchestrator's run is now complete.

## What this phase does NOT do

- Does not ask "is this good?" / "pass / partial / fail?".
- Does not infer failure causes.
- Does not suggest fixes, follow-up runs, hyperparameter changes, dataset changes.
- Does not write any verdict to `run-summary.md` or elsewhere.
- Does not delete or modify training checkpoints.
- Does not push to any remote / cloud / registry.

The user looks at the videos and makes their own call. If they want to iterate, they re-invoke the orchestrator with a new run-name.

## Failure modes

- **Final checkpoint missing.** Surface and stop. Don't render against an intermediate checkpoint without explicit user consent.
- **`load_checkpoint` OOM at inference time.** Lower `validation.video_dims` in the eval config (smaller renders are still useful for a sanity look). Retry once. If still OOM, surface the failure and let the user run inference manually via `packages/ltx-pipelines/`.
- **All renders look broken/black.** May be an inference-pipeline-side issue rather than a training failure. Mention in the output block: *"If renders look broken across all categories, try `packages/ltx-pipelines/` directly to rule out a pipeline issue."* Then exit. Do not investigate further.

## Do not

- Do not skip Category 1 or 2. They're cheap and informative.
- Do not invent a held-out set if `holdout.jsonl` is missing — the prepare-dataset step decides that.
- Do not coach the user on what "good" means for their use case.
