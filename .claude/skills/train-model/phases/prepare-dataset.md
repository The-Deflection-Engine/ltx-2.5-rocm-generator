# Phase 5 — Prepare Dataset

Procedure document for the `train-model` orchestrator (Phase 5). Read this file in full before acting on the prepare-dataset phase.

Goal: produce a captioned, complete dataset metadata file at `<workspace>/<run-name>/dataset/dataset.json` consumable by `process_dataset.py`. Idempotent — re-runs skip work already done.

The orchestrator's hard invariants apply (see `../SKILL.md`), especially: **no file mutation outside the workspace without explicit user approval.**

## Inputs

The orchestrator passes:
- Source path (directory of videos / single video / pre-existing metadata file).
- Target mode (T2V, I2V, V2V IC-LoRA, V2A, etc.) — determines which columns are required.
- Captioner backend choice (Qwen3-Omni local vLLM server / Gemini Flash cloud / skip).
- Workspace path `<workspace>/<run-name>/`.

## Required Columns by Mode

`process_dataset.py` detects columns by convention and resolves each to a role. The media column may be `video` **or** `audio`. When a dataset has a `video` column with an audio track and no separate `audio` column, audio is **auto-extracted** from the video (unless `--skip-audio`), so an explicit `audio` column is only needed when the audio lives in separate files.

**Video-generating modes** (need a `video` column):

| Mode | Required | Optional |
|------|----------|----------|
| T2V, I2V, video extension/suffix | `video`, `caption` | `audio` (else auto-extracted) |
| Video outpainting | `video`, `caption` | |
| Video inpainting | `video`, `caption`, `video_mask` | |
| V2A (foley) | `video`, `caption` | `audio` (target; else auto-extracted) |
| A2V | `video`, `caption` | `audio` (else auto-extracted from video) |
| V2V IC-LoRA | `video`, `caption`, `reference_video` | |
| AV2AV IC-LoRA | `video`, `caption`, `reference_video`, `reference_audio` | `audio` (else auto-extracted) |

**Audio-only modes** (no `video` column — the media column is `audio`):

| Mode | Required | Optional |
|------|----------|----------|
| T2A | `audio`, `caption` | |
| Audio extension/suffix | `audio`, `caption` | |
| Audio inpainting | `audio`, `caption`, `audio_mask` | |
| A2A IC-LoRA | `audio`, `caption`, `reference_audio` | |

Aliases: `media_path` for `video`, `ref_media_path` for `reference_video`.

## Workflow

### Step 1 — Classify source

```bash
# If source is a file, identify type:
file "<source>"
# If source is a directory, count media:
find "<source>" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.mov" -o -name "*.webm" \) | wc -l
```

Cases:
- **Pre-existing metadata file** (CSV/JSON/JSONL) → copy to `<workspace>/<run-name>/dataset/dataset.json`, audit columns. Skip to Step 4.
- **Directory of short scenes** → skip Step 2, go to Step 3.
- **Directory containing long videos** → run Step 2.
- **Single long video** → run Step 2.

**Stage the media under `dataset/` before captioning — don't discover this by failing.** Both `caption_videos.py` and `process_dataset.py` reference media by paths **relative to the metadata file's own directory**, so the media must live under `<workspace>/<run-name>/dataset/`. Stage it up front into `dataset/videos/` (and write metadata paths relative to `dataset/`, e.g. `videos/1.mp4`):
- Prefer **symlinks** (instant, no disk cost): `ln -s <abs-source>/<clip> <workspace>/<run-name>/dataset/videos/<clip>`. Symlinks pointing at the original source location work correctly.
- Use a **copy** instead if the workspace and source are on different filesystems or the source may move/change.
- Never caption or preprocess directly against an out-of-tree source path (e.g. `/path/to/source-videos`) — it will fail the relative-path resolution. The original source is left untouched either way.
(Scene-split output in Step 2 already lands under `dataset/scenes/`, which satisfies this.)

### Step 2 — Scene splitting (only for long videos)

`split_scenes.py` takes **one video file** at a time (`video_path` and `output_dir` are both positional arguments). When the source is a directory of long videos, iterate over each file. To drop scenes shorter than 2 seconds, use `--filter-shorter-than 2s` (the `--min-scene-length` option is an integer **frame** count, not seconds — don't pass a float).

```bash
cd packages/ltx-trainer
# Single file:
uv run python scripts/split_scenes.py \
  "<video-file>" \
  "<workspace>/<run-name>/dataset/scenes" \
  --filter-shorter-than 2s

# Directory of long videos — iterate:
for f in "<source>"/*.mp4 "<source>"/*.mov "<source>"/*.webm; do
  [ -e "$f" ] || continue
  uv run python scripts/split_scenes.py "$f" \
    "<workspace>/<run-name>/dataset/scenes" \
    --filter-shorter-than 2s
done
```

Result: scenes saved to `<workspace>/<run-name>/dataset/scenes/`. Pass that directory to Step 3.

### Step 3 — Captioning

Skip entirely if a metadata file with all required `caption` entries already exists.

**Use the captioner's default instruction.** `caption_videos.py` ships a well-tuned default caption prompt — use it as-is (do **not** pass `--instruction`). Captioning runs in two phases: a small **spot-check pass** so the user can confirm the captions look sane, then a **full pass** on the rest.

A custom `--instruction` is the exception, not the norm. Only use one when:
- the **nature of the dataset genuinely demands it** (e.g. a narrow domain the default prompt won't describe well), or
- the **user, after seeing the spot-check captions, explicitly asks** for a change (e.g. "too much background detail").

Do not invent a custom instruction pre-emptively, and in particular **do not bake a subject name / trigger word into the captions via `--instruction`** — the trigger word is handled separately at preprocessing (see "Trigger word" below).

#### Choosing a backend

Two backends, with very different hardware needs:

- **`qwen_omni` (local, default):** Qwen3-Omni-30B-A3B-Thinking served by a local vLLM HTTP server (`serve_captioner.py`). ~65 GiB model download. Default **FP8** quantization uses ~31 GiB of weights and **fits on a 40 GiB GPU** (plus KV cache); **bf16** uses ~60 GiB and needs **≥66 GiB free VRAM**.
- **`gemini_flash` (cloud):** Google `gemini-3.5-flash`. No local model, runs anywhere, parallelisable with `--num-workers`.

**Steer modest hardware to Gemini.** If the GPU is below ~40 GiB (i.e. typical consumer cards — 24 GB / 32 GB), it can't host even the FP8 server, so the local captioner isn't an option — recommend `gemini_flash` and tell the user they'll need Gemini auth: either a `GEMINI_API_KEY`/`GOOGLE_API_KEY` (get one at <https://aistudio.google.com/apikey>) or working gcloud/Vertex AI credentials. If they can't or won't set that up and the hardware can't run Qwen3, the only remaining path is bringing their own captions in the dataset metadata (skip captioning entirely). On a 40 GiB+ GPU the local server is viable (FP8); bf16 needs an 80GB-class card.

#### Qwen server prerequisite (qwen_omni only)

The local backend talks to a vLLM server that must already be running. Launch it once in a **separate terminal** (it stays loaded across captioning runs):

```bash
cd packages/ltx-trainer
uv run python scripts/serve_captioner.py        # FP8 by default, serves on http://127.0.0.1:8001/v1
# bf16 (needs >= 66 GiB free VRAM):  --quantization bf16
# different port/interface:           --port 9000 --host 0.0.0.0
```

First launch downloads the model (~65 GiB). `caption_videos.py` reaches it via `--vllm-url` (default `http://127.0.0.1:8001/v1`). Skip this entirely when using `gemini_flash`.

#### 3a — Spot-check pass (3 samples)

Caption 3 samples with the **default prompt** (no `--instruction`) to confirm the captioner is producing sane output before committing to the whole set.

```bash
cd packages/ltx-trainer
# qwen_omni (server from the previous step must be running):
uv run python scripts/caption_videos.py \
  "<workspace>/<run-name>/dataset/videos/<one-staged-clip>" \
  --output "<workspace>/<run-name>/dataset/preview-captions.json" \
  --captioner-type qwen_omni
  # Point at 3 staged clips under dataset/videos/ (a small subdir or 3 explicit files) — not the out-of-tree source.
  # Optional: --vllm-url http://127.0.0.1:9000/v1   (if the server uses a non-default port)
```

Print the 3 captions **in full** to the user, then **STOP and wait** for their explicit verdict:

> "Here are sample captions from the default prompt. Please review them — reply 'good' to caption the rest, or tell me what to change."

**This is a hard gate. Do NOT caption the full set until the user explicitly approves the samples.** Do not auto-proceed, do not assume "looks fine," do not batch this with other questions. The user must either approve or give tuning instructions first — the whole point of the spot-check is to let them judge caption quality and content before paying for the full pass.

If the user requests changes, introduce a custom `--instruction` (or switch backend), re-run the spot-check on the same 3 samples, show the new captions, and **stop for approval again**. Loop until the user approves. If a custom instruction still isn't converging after a few rounds, switch captioner backend or have the user supply a few manual captions as examples — but still don't proceed to the full set without their OK.

#### 3b — Full pass

Run on the **staged media dir** (`dataset/videos/` from Step 1, or `dataset/scenes/` from Step 2) with the **default prompt** (or the same `--instruction` only if one was explicitly agreed in 3a). Caption the staged in-tree media — not the original out-of-tree source path.

**Qwen3-Omni (local — server must be running):**

```bash
cd packages/ltx-trainer
uv run python scripts/caption_videos.py \
  "<workspace>/<run-name>/dataset/videos" \
  --output "<workspace>/<run-name>/dataset/dataset.json" \
  --captioner-type qwen_omni
```

**Gemini Flash (cloud — runs anywhere, parallelisable):**

```bash
# Auth: GEMINI_API_KEY / GOOGLE_API_KEY env var, or gcloud / Vertex AI credentials.
cd packages/ltx-trainer
uv run python scripts/caption_videos.py \
  "<workspace>/<run-name>/dataset/videos" \
  --output "<workspace>/<run-name>/dataset/dataset.json" \
  --captioner-type gemini_flash \
  --num-workers 5
```

Output: JSON list of `{caption, media_path}` with paths **relative to the output file location**. The 3 spot-check captions can be merged in to avoid re-captioning them.

#### Trigger word (handled at preprocessing, not in captions)

For style/concept LoRAs the trigger word is **not** written into the captions here. It is prepended to every caption at preprocessing: pass `--lora-trigger "<word>"` to `process_dataset.py` in Phase 7, which forwards it to the caption-processing step (`process_captions.py`, where the prepend actually happens). That is the canonical mechanism — keep the captions describing what's actually on screen (via the default prompt), and let the trigger flag bind the concept to the token. Record the chosen trigger word in the plan so Phase 7 passes it through. Do not also bake the word into captions (it would double up).

**The injection mechanism is a fixed implementation detail — never make it a user-facing question.** The *only* trigger-word thing to ask the user is the **word itself** (or whether they want a trigger word at all). Do **not** ask, mention, or present as an option *how* it gets injected (e.g. "inject into the caption vs via `process_dataset`") — it is always `--lora-trigger`, full stop. Surfacing the method as a choice creates unnecessary confusion.

### Step 4 — Conditioning inputs (modes that need references or masks)

Some modes need a per-sample conditioning input beyond the video/audio and caption:

| Mode | Required extra input | Column |
|------|----------------------|--------|
| V2V IC-LoRA | reference video | `reference_video` (alias `ref_media_path`) |
| AV2AV IC-LoRA | reference video + reference audio | `reference_video`, `reference_audio` |
| A2A IC-LoRA | reference audio | `reference_audio` |
| Video inpainting | per-frame video mask | `video_mask` |
| Audio inpainting | audio mask | `audio_mask` |

These inputs encode **the user's specific idea** for the LoRA (what the reference represents, which regions the mask covers). There is no universal recipe, so **do not invent or default to a particular method** (e.g. don't assume Canny edges, depth, pose, or some generic box/border mask). The agent must not pick the conditioning semantics for the user.

Workflow when the chosen mode needs one of these and the dataset doesn't already provide it:

1. **Check first** — if the user already supplied the column (and the files exist), use it as-is and move on.
2. **Otherwise, ask the user to provide it**, explaining concretely what's needed: the column name, that it's one file per sample aligned to each clip, and that the *content/semantics are their call* (what the reference should depict, what the mask should cover). Make clear this reflects their specific use-case — you won't guess it.
3. **Help generate only if the user asks.** If they say "can you generate the references/masks by doing X" (X = their described method), then help: write or run a small script for *their* approach, or use a repo tool if it fits. One such tool exists — `scripts/compute_reference.py` generates **Canny edge** reference videos — but only mention/use it if the user specifically wants Canny; never offer it as the default.
4. **Hard gate:** do not proceed to preprocessing for a conditioning mode until the required column is present with real files. Surface clearly if it's missing.

**Column-naming note (if references are generated):** `compute_reference.py` writes a `reference_video` field, which
`process_dataset.py` detects automatically. Legacy datasets using `ref_media_path` also work.

### Step 5 — Holdout split

Reserve a subset of samples as a **held-out set** never seen during training. This is what Phase 9 (post-train validate) renders against to test true generalization rather than memorization.

Decision tree:

1. **User already supplied a held-out set** (separate file or directory they explicitly nominated): do not split. Copy/reference their file to `<workspace>/<run-name>/dataset/holdout.jsonl` and leave `dataset.json` as-is.
2. **Small dataset** (judge qualitatively; tens of samples or fewer): holding samples out meaningfully reduces training capacity. Ask the user:
   > "Dataset has <N> samples. Reserving a holdout meaningfully reduces what's available for training. Options: (a) reserve 1–2 for holdout, (b) skip holdout — post-train eval will only test in-distribution. Your call."
3. **Otherwise:** auto-split. Reserve a small fraction (this skill's default: roughly 10% of samples, bounded so the holdout doesn't grow huge — a handful of held-out samples is usually enough). Use seed 42 for the split so it's reproducible. Surface the count and the picked IDs in the plan.

After splitting, write `<workspace>/<run-name>/dataset/holdout.jsonl` (one JSON object per line with the same columns as `dataset.json`). **Remove the held-out entries from `dataset.json`** so they don't enter preprocessing or training.

Always print:
> "Held out <K> of <N> samples for post-train evaluation. Held-out IDs: <list>."

If holdout is skipped, surface in the plan: *"Skipping holdout — dataset is too small. Post-train eval will only render in-distribution prompts; true generalization isn't testable for this run."*

### Step 6 — Audit

Before returning to the orchestrator, verify the metadata file has all required columns for the chosen mode. Print a one-line summary:

> "Prepared <N> training samples (+ <K> held out) for <mode>. Columns: <list>. Saved to `<workspace>/<run-name>/dataset/dataset.json` (+ `holdout.jsonl`)."

## Idempotency

- If `dataset.json` already exists and all required columns are present: skip captioning. Confirm reuse with the user only if the file was supplied by them outside the workspace (per the orchestrator's file-safety invariant).
- If captioning was partial (some entries missing `caption`), re-run captioning only on the missing entries by filtering the metadata file before passing to `caption_videos.py`.

## Failure Modes

- Qwen server won't start / OOMs on launch → use the default `--quantization fp8` (not `bf16`), lower `--gpu-memory-utilization`, or reduce `--max-model-len` on `serve_captioner.py`. If the GPU simply can't host a 30B model, switch to `gemini_flash`.
- `caption_videos.py` can't connect (qwen_omni) → the vLLM server isn't running or `--vllm-url` is wrong. Start `serve_captioner.py` first and confirm the URL/port match.
- Gemini rate-limit → reduce `--num-workers`, retry.
- Scene splitter produces 0 scenes → the detector found no cuts, or `--filter-shorter-than` removed everything. Lower/remove `--filter-shorter-than`, or adjust the detector threshold. (`--min-scene-length` is an integer minimum-frames-per-scene for the detector, not a short-scene filter.)
- IC-LoRA reference compute fails on some frames → script logs the failures; report counts to the user and ask whether to proceed with the remaining samples or stop.

## Do Not

- Do not move or rename the user's source files. The skill reads them in place; the workspace contains only **derived** artifacts.
- Do not delete `scenes/` or partial captioning outputs without approval — they may be expensive to regenerate.
