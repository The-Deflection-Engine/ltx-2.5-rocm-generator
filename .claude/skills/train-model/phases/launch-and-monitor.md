# Phase 8 — Launch & Monitor

Procedure document for the `train-model` orchestrator (Phase 8 + monitor-only re-entry). Read this file in full before acting on launch/monitor.

Goal: start the training job, surface the W&B URL, produce periodic status reports, write `run-summary.md` at completion.

The orchestrator's hard invariants apply (see `../SKILL.md`).

## Launch — Single GPU

```bash
cd packages/ltx-trainer
uv run python scripts/train.py "<workspace>/<run-name>/config.yaml"
```

## Launch — Multi-GPU

Use Accelerate. For LoRA, DDP (default) is fine. For full FT, use FSDP.

```bash
cd packages/ltx-trainer

# DDP (LoRA, multi-GPU)
uv run accelerate launch scripts/train.py "<workspace>/<run-name>/config.yaml"

# FSDP (full FT, multi-GPU)
uv run accelerate launch \
  --config_file configs/accelerate/fsdp.yaml \
  scripts/train.py "<workspace>/<run-name>/config.yaml"
```

**Pass `--disable-progress-bars` to `train.py` whenever stdout is redirected to a log file** (every background run) or running multi-GPU. The Rich progress bar rewrites a single line with carriage returns and does **not** flush parseable newlines to a redirected log, so without this flag the log shows no step/loss lines and you're forced to poll `nvidia-smi`. With it, step/loss lines are written normally and the log is greppable.

## Pre-Launch Checks (every launch)

1. Run the self-check from `references/config-patching.md` (paths exist, frame/resolution constraints, generated modalities have matching latents dirs, references/masks dirs present for conditional modes).
2. Confirm `nvidia-smi` shows expected GPUs available (not occupied by another process).
3. If `wandb.enabled: true`, confirm credentials still resolve: `uv run python -c "import wandb; print(bool(wandb.Api().api_key))"` → `True`. (Don't use `wandb status`.) If `False`, surface to the user before launching — they may want to `wandb login` or run without tracking.

## Run In Background, Monitor Foreground

Long training runs should not block the agent's response loop, and they must survive past the launching turn.

**Prefer the agent's managed/native background-shell mechanism** (the harness facility for long-running background commands — output streaming + PID/exit tracking, survives across turns). It's the reliable way to launch training: it stays alive, streams to a log the agent can poll, and reports completion. Launch the training command through that mechanism, writing to `<workspace>/<run-name>/logs/train.log` with `--disable-progress-bars`, using **absolute paths** for the config and log (`uv run --directory packages/ltx-trainer` changes the cwd, so a relative config path won't resolve).

**`nohup ... &` is a last-resort fallback only.** Detached jobs can be harder to track and may not survive environment/session cleanup, so only use it if no managed background mechanism is available, and verify the PID is still alive afterward:

```bash
# Fallback ONLY — prefer the managed background shell above.
mkdir -p "<workspace>/<run-name>/logs"
nohup uv run --directory packages/ltx-trainer python scripts/train.py \
  "<ABSOLUTE path>/<run-name>/config.yaml" --disable-progress-bars \
  > "<ABSOLUTE path>/<run-name>/logs/train.log" 2>&1 &
echo $! > "<workspace>/<run-name>/logs/train.pid"
```

## Status Report

Produce on user request (or every <interval> automatically). Pull from:

- **W&B run URL:** First lines of `train.log` after init, or `wandb.run.url` from a `wandb` Python snippet. Surface as a clickable URL.
- **Latest step:** `tail -n 200 "<workspace>/<run-name>/logs/train.log" | grep -oE "step [0-9]+" | tail -1`.
- **Recent loss:** `tail -n 200 "<workspace>/<run-name>/logs/train.log" | grep -oE "loss[: ]+[0-9.]+" | tail -5`.
- **Checkpoints saved:** `ls -1t "<workspace>/<run-name>/outputs/checkpoints/" 2>/dev/null`.
- **Validation samples:** `ls -1t "<workspace>/<run-name>/outputs/samples/" 2>/dev/null`.
- **GPU utilization snapshot:** `nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader`.
- **ETA:** only report one **grounded in real numbers the trainer has actually emitted** — do not invent or "educated-guess" an ETA before the trainer has produced per-step timings. Compute as `(total_steps - current_step) * recent_avg_step_time`, where `recent_avg_step_time` is measured from **steady-state training steps** (the trainer's reported per-step time or log timestamps), **excluding** one-time setup that doesn't repeat per step: model loading, the step-0 validation pass, and periodic validation passes. The trainer's own early ETA projection is skewed high by the slow step-0 validation and settles after a few steps — wait for it to settle rather than quoting the inflated early figure. Until real step timings exist, say "measuring step time…" rather than guessing a duration.

Format the report tightly — one block, no fluff:

```
Step 1240 / 2000 (62%) — loss ~0.072 (last 5: 0.071, 0.073, 0.069, 0.075, 0.072)
ETA: ~1h 24m  |  GPU: 91% util, 39.2 / 48.0 GB
Latest checkpoint: lora_weights_step_01000.safetensors
Latest validation: samples/step_01200_*.mp4
W&B: https://wandb.ai/<entity>/<project>/runs/<id>
```

## Monitor-Only Mode

Invoked when the orchestrator detects an existing `<workspace>/<run-name>/` with checkpoints or a running process.

1. Check if the training process is live: `[ -f .../logs/train.pid ] && kill -0 $(cat .../logs/train.pid) 2>/dev/null && echo LIVE || echo STOPPED`.
2. Produce the same status report as above.
3. If STOPPED:
   - Compute step from last checkpoint.
   - Find the latest checkpoint pair (`lora_weights_step_*.safetensors` or `model_weights_step_*.safetensors`, plus a matching `training_state_step_*.pt` when resume state is enabled) under `<workspace>/<run-name>/outputs/checkpoints/`.
   - Patch `model.load_checkpoint` in `config.yaml` to point at that checkpoint file (this is the only way the trainer knows to resume — there's no auto-detection from `output_dir`).
   - Surface the resume command: `uv run python scripts/train.py "<workspace>/<run-name>/config.yaml"`.
   - Ask the user to confirm both the patch and the launch before applying.

## Resume

The trainer **does not auto-resume from `output_dir`**. To resume an interrupted run:

1. Set `model.load_checkpoint` in `config.yaml` to the latest checkpoint file (e.g. `<workspace>/<run-name>/outputs/checkpoints/lora_weights_step_02000.safetensors`).
2. Launch normally. The trainer loads those weights, then looks for a matching `training_state_step_*.pt` **next to the loaded checkpoint** and restores optimizer/scheduler/step state from it. If the state file is missing, weights load but training starts from step 0.
3. To load the weights but skip the state restore (e.g. for branching off into a new run from a known-good checkpoint), set `checkpoints.no_resume: true`.

Always ask the user before patching `model.load_checkpoint` or setting `no_resume` — checkpoints are precious.

## Failure During Training

If the training process exits non-zero:

1. Tail the log and identify the error type.
2. Cross-reference `references/troubleshooting.md`.
3. Propose a config fix (with the exact diff to `config.yaml`).
4. Ask the user before applying. Then resume with the patched config.

Never silently restart a failed training run without acknowledging the failure to the user.

## After Training Completes

1. Write a **run summary** to `<workspace>/<run-name>/outputs/run-summary.md` (see below) so the user can find their bearings months later without rereading the trainer docs.
2. Show final checkpoint path and step count.
3. Show W&B URL if enabled.
4. Return control to the orchestrator (Phase 9 — post-train validate runs next).

### Writing `run-summary.md`

The summary is the **landing page** for this run. Anyone (including the user months from now) should be able to read it and understand: what was trained, on what data, with what config, where everything lives, how to use the result, and how to continue. The trainer doesn't produce this — the skill does.

Template (fill from `plan.md`, `config.yaml`, `autotune.log`, dataset metadata, training log):

```markdown
# <run-name>

**Trained:** <YYYY-MM-DD HH:MM> on <GPU(s)>
**Final checkpoint:** `outputs/checkpoints/<filename>.safetensors` (step <N>)

## What this LoRA does

<One paragraph from the plan's Goal section — restating the user's intent.>

## Trigger word

`<trigger>` — include in prompts at inference time. (Omit this section if no trigger word.)

## Mode

<Mode name> (<lora|full>). Conditioning: <list of conditions, or "none">.

## Dataset

- Source: `<absolute path>`
- Captioning: <`qwen_omni` (Qwen3-Omni-30B via vLLM) | `gemini_flash` | "user-supplied">
  - Captioner instruction used: <verbatim string, or "default">
- Samples: <N training> + <K held-out> at <W>x<H>x<F>
- Preprocessed to: `dataset/.precomputed/`

## Training config

Final values after autotune (deltas from baseline noted in `autotune.log`):

| Field | Value |
|-------|-------|
| Optimizer | <...> |
| Mixed precision | <bf16/fp16> |
| Quantization | <...> |
| Gradient checkpointing | <on/off> |
| Batch size × grad accum | <B> × <A> (effective <BxA>) |
| LoRA rank / alpha | <R> / <A> (or "full FT") |
| LoRA target modules | <list> (or "n/a") |
| Steps | <N> |
| Learning rate | <value> |
| Step time (final) | ~<T>s |
| Peak VRAM | ~<V> GB |

Full config: `<workspace>/<run-name>/config.yaml`

## Outputs

- Checkpoints: `outputs/checkpoints/`
- Validation samples (during training): `outputs/samples/`
- Post-train eval renders (if Phase 9 ran): `outputs/eval/`
- W&B run: <url, or "(W&B not enabled)">

## How to use this checkpoint

For inference, point `packages/ltx-pipelines/` at the final checkpoint. Example invocation:

\`\`\`bash
# (Minimal sketch — adapt to the pipeline you're using.)
# load base LTX-2 model + apply this LoRA from outputs/checkpoints/<filename>.safetensors
\`\`\`

## How to continue training

To resume from the final checkpoint (e.g. more steps, different LR), edit `config.yaml`:

\`\`\`yaml
model:
  load_checkpoint: "<absolute path to outputs/checkpoints/<filename>.safetensors>"
optimization:
  steps: <new total>          # trainer resumes optimizer/scheduler/step from the training_state_step_*.pt sitting next to the checkpoint above
\`\`\`

Then re-launch with the same command in the "Launched with" section below.

## How this was launched

\`\`\`
<exact command used, with the workspace's absolute config path>
\`\`\`

## Reproducibility

- Seed: <value>
- Workspace: `<absolute path>`
- Repo commit at launch: `<git rev-parse HEAD output>`
- LTX-2 model: `<model.model_path from config>`
- Text encoder: `<model.text_encoder_path from config>`
```

Write the file using the `Write` tool. Don't embed it in a heredoc — the markdown nested in this skill is illustrative; fill the template with real values from the run's artifacts.

### Next steps (surface to user)

After writing the summary, point the user at:
- The summary file path.
- The final checkpoint path.
- The W&B URL if enabled.
- The upcoming Phase 9 (post-train validate) — the orchestrator handles the transition.

Suggest, but don't run:
- Test inference with `packages/ltx-pipelines/`.
- Push to HF Hub via the trainer's `hub.push_to_hub` config (a separate, lightweight re-launch).
- Continue training from the final checkpoint (see summary's "How to continue training" section).

## Do Not

- Do not modify `output_dir` contents after a run completes — checkpoints belong to the user now.
- Do not start a second training run into the same `output_dir` without explicit user approval. Resume requires patching `model.load_checkpoint` (the trainer does not auto-detect prior checkpoints); a true fresh-start from the same dir additionally needs `checkpoints.no_resume: true`.
- Do not auto-restart a failed run without diagnosis and user approval.
