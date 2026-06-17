# Phases 6 (one-sample) & 7 (full) — Preprocess Dataset

Procedure document for the `train-model` orchestrator. Read this file in full before acting on the preprocess phase.

Goal: run `process_dataset.py` to produce VAE latents, audio latents, and text embeddings. Two modes:
1. **One-sample** (Phase 6 sanity check) — preprocess a single sample to `<workspace>/<run-name>/overfit/.precomputed/`.
2. **Full** (Phase 7) — preprocess the whole dataset to `<workspace>/<run-name>/dataset/.precomputed/`.

The orchestrator's hard invariants apply (see `../SKILL.md`), especially: **never silently overwrite existing user data.**

## Required Subdirectories by Mode

Under `.precomputed/`:

| Subdir | Required for |
|--------|--------------|
| `latents/` | Video-bearing modes only (T2V, I2V, video extend/inpaint/outpaint, V2V/AV2AV IC-LoRA, A2V, V2A). **Not** produced for audio-only modes. |
| `conditions/` | Always (text embeddings) |
| `audio_latents/` | Any mode with audio: video modes carrying audio, plus all audio-only modes (T2A, audio extend/suffix/inpaint, A2A IC-LoRA) |
| `reference_latents/` | V2V IC-LoRA, AV2AV IC-LoRA |
| `reference_audio_latents/` | A2A IC-LoRA, AV2AV IC-LoRA |
| `video_masks/` | Video inpainting |
| `audio_masks/` | Audio inpainting |

**Audio-only modes** (T2A, audio extend/suffix, audio inpainting, A2A IC-LoRA) produce `audio_latents/` + `conditions/` (plus `audio_masks/` or `reference_audio_latents/` as applicable) and **no `latents/`**. Do not flag a missing `latents/` as incomplete for these modes.

## Workflow — Full Preprocess

### Step 1 — Verify existing `.precomputed/` (if present)

If `<workspace>/<run-name>/dataset/.precomputed/` already exists:

1. List subdirectories present. Confirm all required for the chosen mode are present.
2. Load one sample per modality and check tensor shapes:
   ```bash
   uv run python -c "import torch; t = torch.load('<path>'); print(t.shape if hasattr(t, 'shape') else {k: v.shape for k, v in t.items()})"
   ```
3. Compare shapes against the target resolution from the plan.

**On any mismatch or missing subdirectory: STOP. Do not run `process_dataset.py`.** Ask the user via `AskUserQuestion`:
- Reuse the existing data at its current resolution (update plan + config accordingly).
- Re-preprocess to a new directory (`<workspace>/<run-name>/dataset/.precomputed-v2/` etc.) — preserves the existing data.
- Abort.

**Never pass `--overwrite` without explicit user approval** for this exact action.

### Step 2 — Invoke `process_dataset.py`

```bash
cd packages/ltx-trainer
uv run python scripts/process_dataset.py \
  "<workspace>/<run-name>/dataset/dataset.json" \
  --resolution-buckets "<W>x<H>x<F>" \
  --model-path "<absolute-model-path>" \
  --text-encoder-path "<absolute-gemma-path>" \
  --output-dir "<workspace>/<run-name>/dataset/.precomputed" \
  --load-text-encoder-in-8bit         # on 32GB tier (low-VRAM config), per t2v_lora_low_vram.yaml
```

Add as needed:
- `--skip-audio` — if mode doesn't use audio (T2V video-only variants).
- `--audio-durations "<list>"` — for T2A from a captions-only file.
- `--lora-trigger "<trigger>"` — for style/concept LoRAs.
- `--reference-downscale-factor <N>` — for IC-LoRA modes if downscaled references are desired.
- `--video-column`, `--caption-column` — only if the metadata file uses non-standard column names.

**Do not pass `--overwrite`** unless re-preprocessing was explicitly approved in Step 1.

### Step 3 — Audit output

After completion, verify:

```bash
ls "<workspace>/<run-name>/dataset/.precomputed/"
# Expected subdirs per the mode table above.

# Count files in each:
for d in latents conditions audio_latents reference_latents video_masks audio_masks; do
  if [ -d "<workspace>/<run-name>/dataset/.precomputed/$d" ]; then
    echo "$d: $(ls "<workspace>/<run-name>/dataset/.precomputed/$d" | wc -l)"
  fi
done
```

Counts in each required subdir should equal the dataset sample count.

**Reconcile counts — do this for every run, any dataset size.** Compare the `latents/` (and `audio_latents/`) count against the caption/sample count. If fewer latents were produced, `process_dataset.py` **silently skipped** clips — most commonly because they were **shorter than the target frame bucket** (it logs each skip). When counts don't match:

1. Identify which clips were dropped (grep the preprocess log for skip/"fewer frames" lines, or diff the produced `.pt` stems against the metadata).
2. **Surface it to the user** with the count and the specific clips — never silently proceed on a shrunk dataset.
3. Offer options: re-preprocess at a **smaller frame bucket** the clips support, add a **second (shorter) bucket** to keep the short clips (multi-bucket requires `batch_size: 1`), or accept the loss. Let the user decide.

**Audio gate (hard stop for audio runs).** For any run with an audio modality (joint audio+video, A2V, V2A, T2A, audio-only modes), verify `audio_latents/` is **present and non-empty** with one `.pt` per sample. `process_dataset.py` **swallows audio-decode errors and continues** — it logs "0 videos with audio" and produces empty `audio_latents/` rather than failing. If an audio run produced no audio latents, **stop** — do not proceed to training (it would silently train audio-free). The usual cause is a broken audio decode path (e.g. missing/incompatible `torchcodec`); confirm `uv run python -c "import torchaudio; torchaudio.load('<a clip>')"` works (see `references/troubleshooting.md`), fix it, then re-preprocess with `--overwrite`.

## Workflow — One-Sample (Phase 6)

Same as full preprocess, but operate on a single-sample metadata file:

1. Pick the first sample from `<workspace>/<run-name>/dataset/dataset.json` and write a one-sample metadata file **inside the dataset dir** — e.g. `<workspace>/<run-name>/dataset/_one_sample.json` — copying the entry **verbatim, keeping its relative `media_path`**. `process_dataset.py` resolves media paths relative to the metadata file's own directory, so the one-sample file must sit beside the real media (i.e. in `dataset/`, the same dir as `dataset.json`). **Do not** place it in `overfit/` and **do not** rewrite the path to an absolute one — an absolute path produces mirrored nested output dirs (`.precomputed/latents/absolute/path/.../x.pt`) instead of a clean `latents/x.pt`.
2. Run `process_dataset.py` on that file with `--output-dir "<workspace>/<run-name>/overfit/.precomputed"` (output still goes to `overfit/`, only the metadata lives in `dataset/`).
3. Use the **same `--resolution-buckets`** as the planned full run. Critical: a small-shape sanity check is misleading because resolution is the dominant memory factor.
4. Clean up the temporary `dataset/_one_sample.json` afterward (it's scratch; don't leave it in the dataset dir).

## Decode-and-Verify (optional debug aid)

If the user reports validation samples look wrong or training diverges, decode one preprocessed sample back to media:

`decode_latents.py` takes the **latents directory** and an **output directory** as positional arguments (it decodes the whole directory, not a single `.pt` file). Add `--with-audio` and `--audio-latents-dir` if the dataset has audio.

```bash
cd packages/ltx-trainer
uv run python scripts/decode_latents.py \
  "<workspace>/<run-name>/dataset/.precomputed/latents" \
  "<workspace>/<run-name>/dataset/.precomputed/decoded_check" \
  --model-path "<absolute-model-path>"
```

If decoded output is garbled, preprocessing itself is suspect (wrong model, wrong VAE).

## Failure Modes

- **"shape mismatch" on resume:** Step 1's verification check. Ask user before any mutation.
- **`frames % 8 != 1`** error from process_dataset.py: the requested frame count is invalid; correct in the plan and re-launch.
- **VRAM OOM during preprocessing:** add `--load-text-encoder-in-8bit`. If still OOM, reduce `--batch-size`.
- **Disk full:** preprocessed latents can be large (especially audio). Surface to user with a `du -sh` summary of `.precomputed/`.

## Do Not

- Do not delete or overwrite existing `.precomputed/` data without explicit user approval for that exact action.
- Do not preprocess at a smaller resolution to "save time" — the sanity check exists specifically to validate the planned resolution.
