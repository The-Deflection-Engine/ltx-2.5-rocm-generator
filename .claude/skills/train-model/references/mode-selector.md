# Mode Selector

Map the user's stated intent to a `flexible`-strategy configuration. All modes are supported via a single strategy (`training_strategy.name: "flexible"`); the difference is which modality is generated and which `conditions` are attached.

> Full reference: [`packages/ltx-trainer/docs/training-modes.md`](../../../../packages/ltx-trainer/docs/training-modes.md). This file is the **lookup table** for translating user intent.

## Decision Table

| User says (roughly)... | Mode | Example config | Modalities | Conditions |
|------------------------|------|----------------|------------|------------|
| "generate videos from text", "T2V LoRA" | T2V | `configs/t2v_lora.yaml` | video gen, audio gen | none |
| "generate videos from a starting image", "I2V" | I2V | `configs/i2v_lora.yaml` | video gen, audio gen | `first_frame` (video) |
| **plain concept/style LoRA** ("train a LoRA on X", no specific task) | **I2V by default** (see note) | `configs/i2v_lora.yaml` | video gen, audio gen | `first_frame` (video), `probability: 0.5` |
| "extend a video forward in time" | Video extension (prefix) | `configs/video_extend_lora.yaml` | video gen, audio gen | `prefix` (video) |
| "extend a video backward in time" | Video extension (suffix) | `configs/video_suffix_lora.yaml` | video gen, audio gen | `suffix` (video) |
| "fill in masked regions of a video" | Video inpainting | `configs/video_inpainting_lora.yaml` | video gen | `mask` (video) |
| "expand a video beyond its borders" | Video outpainting | `configs/video_outpainting_lora.yaml` | video gen | `spatial_crop` (video) |
| "style transfer from reference video", "IC-LoRA", "depth/pose/canny control" | V2V IC-LoRA | `configs/v2v_ic_lora.yaml` | video gen | `reference` (video) |
| "generate video to match an audio track" | A2V | `configs/a2v_lora.yaml` | video gen, audio frozen | none (audio `is_generated: false`) |
| "add sound effects to silent video", "foley", "V2A" | V2A | `configs/v2a_lora.yaml` | video frozen, audio gen | none (video `is_generated: false`) |
| "generate audio from text", "T2A" | T2A | `configs/t2a_lora.yaml` | audio gen | none |
| "extend audio forward / backward" | Audio extension | `configs/audio_extend_lora.yaml`, `configs/audio_suffix_lora.yaml` | audio gen | `prefix` / `suffix` (audio) |
| "fill in masked regions of audio" | Audio inpainting | `configs/audio_inpainting_lora.yaml` | audio gen | `mask` (audio) |
| "audio style transfer from reference", "A2A IC-LoRA" | A2A IC-LoRA | `configs/a2a_ic_lora.yaml` | audio gen | `reference` (audio) |
| "joint video+audio reference control" | AV2AV IC-LoRA | `configs/av2av_ic_lora.yaml` | video gen, audio gen | `reference` (both) |
| Any of the above with full fine-tune | Full FT variant | as above, set `model.training_mode: "full"` | (mode-specific) | (mode-specific) |

### Why I2V is the default for a plain concept/style LoRA

A "train a LoRA on X" request isn't tied to one inference mode: **LoRA weights are pipeline-agnostic** — the same checkpoint loads in both T2V and I2V inference (both use `TI2VidOneStagePipeline`/`TwoStages`). The `i2v_lora` config trains `first_frame` with **`probability: 0.5`**, so the model learns both first-frame-conditioned (I2V) and unconditioned (T2V) generation in one run, and the first frame comes from each training clip automatically (no extra data prep). That makes I2V a versatile **superset** — usable for both at inference at no extra cost — which is why it's the default for a plain LoRA. Ask the user how they'll use it (text-only / from an image / both) and only drop to `t2v_lora` if they're sure it's text-only. (See the orchestrator `SKILL.md` Phase 1.)

## Disambiguation Questions

When the user's first answer is ambiguous, ask **one** follow-up:

- "extend a video" → forward or backward in time?
- "control with a reference" → video reference (depth/pose/canny/etc.) or audio reference?
- "fill in regions" → video regions (masked frames) or audio regions (masked time)?
- "T2V" → joint video+audio (default) or video-only?
- LoRA or full fine-tune? Default to LoRA unless the user has multi-GPU + clear reason for full.

## Combining Modes

The flexible strategy allows stacking conditions. Common combinations:

- **I2V + V2A** (start frame + generate audio for the resulting video) — not directly expressible; would need two passes.
- **Video extension + audio extension** — both modalities generate, both have `prefix` condition. Express in one config.
- **IC-LoRA + I2V** — `first_frame` + `reference` conditions on the video modality.

If the user asks for a combination not listed, check `packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py` for which conditions can co-exist on a modality. Audio modality cannot use `first_frame` or `spatial_crop`.

## When the Intent Doesn't Map

If after disambiguation there is no entry in the table and no combination of `flexible` conditions covers the user's request, go to the **Escape Hatch** section in the orchestrator `SKILL.md`. Do not silently pick the closest mode.

## LoRA Rank by Use Case

Once the mode is picked, choose `lora.rank` (and matching `lora.alpha`) based on what the LoRA is supposed to capture. These are starting points; autotune doesn't sweep rank because it's a quality knob, not a step-time one.

| Use case | Suggested rank | Notes |
|----------|----------------|-------|
| Single character, single object, single style | 32–64 | Default for most concept LoRAs. Start at 32; bump to 64 if validation samples underfit. |
| Multi-character world, dense series, complex multi-concept | 96–128 | More capacity for distinguishing several concepts inside one LoRA. |
| Camera move, motion, transition (i.e. behavioural, not visual) | 8–16 | Motion is a thin signal — high ranks just memorise frame content. |
| IC-LoRA control (V2V depth/pose/Canny/etc., A2A audio reference) | 16–32 | Start at 16 for structural control (depth, pose, edges); 24–32 if the reference carries richer style/texture. Video IC-LoRA often lands lower than concept LoRAs. |
| LTX-2 trainer's default if unsure | 32 | Safe baseline. |

On the **32GB tier** (`hardware-profiles.md`), the low-VRAM config already pins `lora.rank: 16`. If the user picks a higher-rank use case on 32GB, surface the trade-off in the plan but let them decide — the autotune sweep doesn't touch rank, so a too-high rank will simply OOM at training time.

Keep `alpha == rank` unless the user has a specific reason otherwise (effective scaling = `alpha / rank`).

## Starting Config

Copy the matching example config — the exact filename from the **Example config** column of the decision table above (e.g. T2V → `configs/t2v_lora.yaml`, I2V → `configs/i2v_lora.yaml`, A2A IC-LoRA → `configs/a2a_ic_lora.yaml`) — into `<workspace>/<run-name>/config.yaml` as the starting point. Then patch per `references/config-patching.md`.
