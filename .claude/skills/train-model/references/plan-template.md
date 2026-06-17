# Plan Template

Write `<workspace>/<run-name>/plan.md` following this template. The plan is the user's contract with the agent — it's the gate before any heavy work runs.

Scale each section to its relevance. Skip subsections that don't apply (e.g., no captioning if data is already captioned), but never collapse "Assumptions" or "Cost/time estimate".

```markdown
# Training Plan — <run-name>

## Goal

<One paragraph restating the user's intent in their own terms.>

## Mode

**<Mode name>** — <one-line rationale linking user intent to mode>.

- Config base: the concrete example config selected by the mode (e.g. `packages/ltx-trainer/configs/t2v_lora.yaml`) — use the actual filename, not a placeholder
- Conditions: <list, or "none">
- Training mode: <`lora` | `full`>

## Dataset

- Source: `<absolute path>` (<N samples>)
- Captions: <`already present` | `will be generated with <backend>`>
- Audio: <`present` | `absent — using --skip-audio` | `to be paired with --audio-durations`>
- IC-LoRA references: <`present` | `to be generated via compute_reference.py` | `n/a`>

## Preprocessing

- Target resolution buckets: `<W>x<H>x<F>` (frames satisfy `frames % 8 == 1`; W,H divisible by 32)
- Estimated time: ~<duration> on detected hardware
- Output: `<workspace>/<run-name>/dataset/.precomputed/`

## Training Config

| Field | Value |
|-------|-------|
| Optimizer | <adamw / adamw8bit> |
| Mixed precision | <bf16 / fp16> |
| Quantization | <null / int8-quanto / ...> |
| Gradient checkpointing | <on / off> |
| Batch size | <N> |
| Gradient accumulation | <N> (effective batch = <N>) |
| Steps | <N> |
| Learning rate | <value> |
| LoRA rank / alpha | <N / N> (or "full FT") |
| LoRA target modules | <list> (or "n/a") |
| LoRA trigger word | <word> (or "n/a") |
| Validation interval | every <N> steps |
| Checkpoint interval | every <N> steps |

## Hardware

- GPU(s): <name> x <count>, <VRAM>GB per GPU
- Launch: <`python scripts/train.py` (single) | `accelerate launch` (multi)>
- VRAM tier: <32GB tier | 40–60GB tier | 80GB+ tier> — <low-VRAM config (`t2v_lora_low_vram.yaml`) | standard config (`t2v_lora.yaml`) | mid-range, autotuned from low-VRAM>

## Sanity Check + Autotune

*In plain terms: before committing to the full run, I do a quick dry run on a single clip at your target resolution to catch out-of-memory or config errors in ~2 minutes (rather than failing hours in), then try a few config variants to pick the fastest one that fits your GPU.*

Mechanics:
- 1 sample, full target resolution, 50 steps + 1 validation pass.
- Autotune sweep: up to 5 trials varying quantization / optimizer / batch size.
- Stops at first OOM or no-improvement.

## Monitoring

<One of:>
- W&B: enabled, project `<name>`, entity `<entity-or-default>`. URL will be surfaced once training starts.
- W&B: **not logged in** — run `wandb login` before training to enable tracking. Otherwise training proceeds without remote logging.

## Outputs

- Training config: `<workspace>/<run-name>/config.yaml`
- Checkpoints: `<workspace>/<run-name>/outputs/checkpoints/`
- Validation samples: `<workspace>/<run-name>/outputs/samples/`
- Autotune log: `<workspace>/<run-name>/autotune.log`

## Assumptions

Defaults the agent chose silently. Override any by replying with the new value.

- <list every non-trivial assumed value: precision, scheduler type, seed, validation prompts, checkpoint retention, etc.>

## Cost / Time Estimate

Give only estimates you can ground; label anything not yet measured as rough. Do **not** state a confident training duration before the sanity check has measured a real step time — say "training duration TBD until the sanity check measures step time" and fill it in afterward (per Hard Invariant #5: no fabricated predictions).

- Captioning: ~<duration> (rough)
- Preprocessing: ~<duration> (rough)
- Sanity check + autotune: ~<duration> (rough)
- Full training: **measured after sanity check** — then `<measured step-time> × <steps>`
- **Total wall-clock estimate:** rough until step time is measured; refine after the sanity check.

## Approve to Proceed

Reply "approve" (or with edits) to start. No captioning, preprocessing, autotune, or training will run before approval.
```

## Notes on Writing the Plan

- Show numbers, not adjectives. "~3 hours" beats "fairly long."
- Surface every assumption that, if wrong, would cost the user time. Better to over-list than under-list — the user can skim.
- If a section reveals you need to ask another question, **stop and ask** before finalizing the plan. The plan is the last gate, not the first.
- If the user's hardware can't reasonably support the requested mode (e.g., single-GPU full FT on a 32GB consumer card), say so plainly in the plan and propose the alternative (LoRA, multi-GPU, etc.), rather than silently downgrading.
