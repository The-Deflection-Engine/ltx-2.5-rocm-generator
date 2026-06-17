# VRAM Tiers

Map probed GPU(s) to a starting training config. The autotune sweep in Phase 6 then empirically improves on this baseline. Use these **tier names** in `plan.md` and user-facing chat — not letter codes.

Source of truth: the two configs shipped in the trainer repo —
`packages/ltx-trainer/configs/t2v_lora.yaml` (standard) and
`packages/ltx-trainer/configs/t2v_lora_low_vram.yaml` (low VRAM).
Per `packages/ltx-trainer/docs/quick-start.md`, the trainer documents
**80GB recommended** and **32GB minimum**. Anything below 32GB is
unsupported by the project.

## Probe

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Pick the **smallest** VRAM tier across the visible GPUs. Multi-GPU only adds throughput at the same per-GPU memory budget — it doesn't relax per-GPU limits.

## Minimum Gate

If per-GPU VRAM is **< 32 GB**, stop the run. Surface to the user:

> "This GPU has <N>GB VRAM. The LTX-2 trainer requires a minimum of 32GB (see `packages/ltx-trainer/docs/quick-start.md`). Training is unlikely to fit even with maximum memory savings, and we don't ship a tested config below 32GB. Options: (a) abort, (b) try anyway with the low-VRAM config and accept it may OOM — purely at your own risk."

Do not invent a sub-32GB tier. The trainer team doesn't ship one.

## 32GB tier — low-VRAM config

**VRAM range:** 32 GB per GPU (trainer minimum).

**Typical GPUs:** RTX 5090, V100 32GB.

Start from `packages/ltx-trainer/configs/t2v_lora_low_vram.yaml` verbatim. Key choices already in that file (do not re-specify in `<workspace>/<run-name>/config.yaml` — copy the file and patch only the paths from `references/config-patching.md`):

- `optimizer_type: "adamw8bit"`
- `enable_gradient_checkpointing: true`
- `batch_size: 1`, `gradient_accumulation_steps: 1`
- `quantization: "int8-quanto"`
- `load_text_encoder_in_8bit: true`
- `offload_optimizer_during_validation: true`
- `lora.rank: 16`, `lora.alpha: 16`

Autotune (Phase 6) will sweep `quantization` off, `optimizer_type` → adamw, and `batch_size` up — but at 32GB the sweep often hits OOM on trial 2 or 3. That's fine; the conservative baseline still works.

## 80GB+ tier — standard config

**VRAM range:** 80 GB per GPU and above (trainer recommended).

**Typical GPUs:** A100 80GB, H100 80GB, H200, B200.

Start from `packages/ltx-trainer/configs/t2v_lora.yaml` verbatim. Key choices already in that file:

- `optimizer_type: "adamw"`
- `enable_gradient_checkpointing: true` (autotune may turn it off if headroom allows)
- `batch_size: 1`, `gradient_accumulation_steps: 1`
- `quantization: null`
- `load_text_encoder_in_8bit: false`
- `lora.rank: 32`, `lora.alpha: 32`

On the 80GB+ tier, the autotune baseline already equals this config (adamw, no quantization), so the quantization/optimizer trials are no-ops; the only real lever is gradient checkpointing off — but for the 22B model that **usually OOMs even with tens of GB of apparent headroom**, so treat a win there as unlikely. The trainer reports its own step-time and peak-VRAM at the end of each run — use those rather than an external timer.

For ≥140GB GPUs (H200, B200), the same 80GB+ tier baseline applies. FA3/FA4 attention backends are viable on Hopper/Blackwell and can speed up training, but they're optional — the trainer's defaults work on PyTorch SDPA without extra setup.

## 40–60GB tier — mid-range (autotune from low-VRAM)

**VRAM range:** 40–60 GB per GPU. The trainer doesn't ship a tested config for this range.

**Typical GPUs:** A40, A6000 48GB, L40, RTX 6000 Ada.

Start from the **32GB tier** (low-VRAM config) and let autotune relax `quantization`, `optimizer_type`, and `batch_size` based on actual headroom. Don't pre-bake intermediate YAML values that haven't been measured. Surface this as **40–60GB tier** in the plan.

## Multi-GPU

If `nvidia-smi` reports N ≥ 2 GPUs of the same model:

- Launch with `uv run accelerate launch scripts/train.py <config>`.
- Use `packages/ltx-trainer/configs/accelerate/fsdp.yaml` for full fine-tune.
- DDP (default `accelerate launch` without a config file) is fine for LoRA.
- Effective batch = `batch_size * gradient_accumulation_steps * num_gpus`. Reduce `gradient_accumulation_steps` proportionally to keep the effective batch consistent with the plan.

## Full Fine-Tune

If the user chose full fine-tune (`model.training_mode: "full"`):

- Require multi-GPU + FSDP on 80GB+ tier GPUs. Otherwise warn in the plan that single-GPU full FT is unlikely to fit and propose LoRA instead.
- Set `acceleration.offload_optimizer_during_validation: true` always (optimizer state is huge under full FT).

## Model Path Constraints

- `model.model_path`: local `.safetensors` only. No URLs.
- `model.text_encoder_path`: local Gemma model directory. No URLs.

If probe didn't find these in conventional locations (`/models/`, `~/models/`, `$LTX_MODELS_DIR`), ask the user in Phase 3 (or offer to download per `references/onboarding.md`).

## Notes on Loss-of-Generality

The two anchor tiers (32GB and 80GB+) correspond directly to the two configs the trainer ships. The autotune sweep is the empirical layer — if a particular GPU consistently lands on a different stable config, **update the relevant trainer config first**, not this file. This skill follows the trainer's choices, not the other way around.
