# First-Run Onboarding

What the orchestrator's Phase 2 probe checks for, what to do when something is missing, and what the skill is allowed to set up automatically (with explicit user approval).

## Prerequisites Checked in Phase 2

| Prerequisite | How to detect | If missing |
|--------------|---------------|------------|
| CUDA GPU visible | `nvidia-smi` returns ≥1 GPU | Stop. Training requires CUDA — point the user at non-LTX-2 docs. |
| Linux | `uname -s` returns `Linux` | Stop. Trainer uses Triton (Linux-only). |
| `uv` installed | `command -v uv` | Offer to install (see "Auto-setup" below). |
| Workspace synced | `[ -f uv.lock ] && uv pip list \| grep -q ltx-trainer` | Offer to run `uv sync` from repo root. |
| LTX-2 model weights | Search `/models/`, `~/models/`, `$LTX_MODELS_DIR` for a `.safetensors` matching `*ltx*2*` | Offer to download (see "Model downloads"). |
| Gemma text encoder dir | Search same locations for a directory containing Gemma config | Offer to download. |
| Captioner backend | Gemini auth (`GEMINI_API_KEY`/`GOOGLE_API_KEY` or gcloud/Vertex), OR a ≥40 GiB GPU to host the Qwen3-Omni-30B vLLM server (FP8), OR captions already in the dataset | See "Captioner graceful degradation" below. **Check the HF cache for an already-downloaded Qwen model before assuming a download is needed** (see note below the table). |
| W&B login (optional) | `uv run python -c "import wandb; print(bool(wandb.Api().api_key))"` → `True` means logged in. Uses wandb's own credential resolution (env/netrc/settings). **Don't** use `wandb status` (reports `api_key: null` even when logged in). | Not a blocker. If `False`: disabled in config + flagged in plan. If the check errors/ambiguous: ask the user, don't assume off. |
| Disk space | `df -h $WORKSPACE` | Surface available space alongside what a run consumes (preprocessed latents, several-GB checkpoints, validation samples). Flag concerns to the user; don't enforce a hard threshold. |

Surface findings as a compact table in chat. For each missing item, present the user with a concrete next step (download command, install command, or "skip this — here's the consequence").

**Check the HF cache before declaring the local captioner unavailable.** The Qwen3-Omni model is served from the HuggingFace cache, not `~/models/`. Before concluding it must be downloaded (or ruling it out on free-disk grounds), check whether it's already cached:

```bash
ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--Qwen--Qwen3-Omni* 2>/dev/null \
  && du -sh "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--Qwen--Qwen3-Omni* 2>/dev/null
```

If it's cached (~60 GiB, all shards present), no download is needed — don't rule out the local captioner because of low free disk on the *home* partition; the weights already exist. Only the GPU-VRAM constraint (≥40 GiB for FP8) then applies.

## Auto-Setup (with explicit user approval)

The skill may, only after the user explicitly says yes, do these setup actions. Each action is a single discrete question.

### Run `uv sync`

```bash
cd <repo-root>
uv sync
```

Ask: *"Repo not synced. Run `uv sync` now? It will download the project's Python dependencies."*

### Install `uv`

Ask: *"`uv` not installed. Install via the official one-liner now?"*

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After install, ask the user to restart their shell or `source ~/.bashrc` before continuing.

### Model downloads

Use `huggingface-cli` (comes with `huggingface-hub`, transitively pulled by `uv sync`). Default destination: `$LTX_MODELS_DIR` if set, else `~/models/` (create if missing). Surface destination in the prompt — never download into the repo or into the workspace.

**LTX-2 base model:**

```bash
huggingface-cli download Lightricks/LTX-2.3 \
  ltx-2.3-22b-dev.safetensors \
  --local-dir ~/models/ltx-2.3
```

Public reference: <https://huggingface.co/Lightricks/LTX-2.3>.

**Gemma text encoder:**

```bash
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir ~/models/gemma-3-12b
```

**Qwen3-Omni captioner (only if the GPU can host it):**

The local captioner is Qwen3-Omni-30B-A3B-Thinking served by a vLLM server (`scripts/serve_captioner.py`), which downloads the model (~65 GiB) on first launch via `uvx vllm` — there's no separate `huggingface-cli` step. Default **FP8** (~31 GiB weights) fits on a **40 GiB** GPU; **bf16** (~60 GiB) needs **≥66 GiB free VRAM**. On a GPU below ~40 GiB (typical consumer 24/32 GB cards), don't use it — use Gemini instead (see "Captioner graceful degradation").

The base model + text encoder are large (multi-GB) downloads. Ask the user before each one — `huggingface-cli` reports the actual size at the start of the transfer. Do **not** batch them into a single "yes/no"; the user may want only what's missing.

### Hugging Face login (if any downloads fail with 401)

Ask: *"Hugging Face download requires login (some Lightricks models are gated). Run `huggingface-cli login` now? You'll need a token from <https://huggingface.co/settings/tokens>."*

## Captioner Graceful Degradation

The captioner is the trickiest prerequisite. The local backend (`qwen_omni`) is now a **30B model served by a vLLM server** — ~65 GiB download; FP8 fits on a 40 GiB GPU, bf16 needs ≥66 GiB. Typical consumer cards (24/32 GB) can't host it, so for most users **prefer Gemini**.

Decision tree:

1. **User already has captions in their dataset metadata** → skip the captioner entirely (Step 3 of `prepare-dataset` skips when captions are present).
2. **GPU ≥40 GiB (FP8) / ≥66 GiB (bf16)** → `qwen_omni` is viable: launch `scripts/serve_captioner.py` first, then caption. Gemini is still fine here too.
3. **GPU below ~40 GiB (the common consumer case)** → `qwen_omni` is not an option. Recommend **`gemini_flash`** and tell the user they need Gemini auth: a `GEMINI_API_KEY`/`GOOGLE_API_KEY` (get one at <https://aistudio.google.com/apikey>) **or** working gcloud/Vertex AI credentials.
4. **No Gemini auth and can't run Qwen3** → the only remaining path is bringing their own captions: add a `caption` column to the dataset metadata, then re-invoke the skill.

Wait for the user's choice. Don't pick one automatically — but make the hardware reality explicit so they don't try to run the server on a card that can't host it.

## What Auto-Setup Does NOT Touch

- Does not modify the user's shell rc files except by explicit instruction (e.g., "add `export LTX_MODELS_DIR=...` to your `.bashrc`?" with the user agreeing).
- Does not modify `~/.gitconfig`, `~/.ssh/`, or any auth-related files.
- Does not install GPU drivers, CUDA, or system packages.
- Does not delete or move existing model files. If a model is found but at an unexpected path, surface and let the user decide.
- Does not download into the repo or workspace. Models live at `~/models/` (or `$LTX_MODELS_DIR`).

## Configuration Inheritance

After downloads, the skill records the resolved paths into the plan's Assumptions section and into `config.yaml`:

```yaml
model:
  model_path: "/home/<user>/models/ltx-2.3/ltx-2.3-22b-dev.safetensors"
  text_encoder_path: "/home/<user>/models/gemma-3-12b"
```

Suggest (don't enforce) setting `LTX_MODELS_DIR=~/models` in their shell rc for future runs.
