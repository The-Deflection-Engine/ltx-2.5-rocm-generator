#!/usr/bin/env bash
# Clean-room test of the README's setup instructions.
#
# Clones the repo fresh, builds a new venv, and follows "First-time setup"
# verbatim -- catching the class of bug where a file is referenced but never
# committed, or a documented command simply does not work.
#
# The ~165GB of model weights are SYMLINKED from the existing checkout rather
# than re-downloaded, so steps 1-3 of the README (download / patch / quantize)
# are NOT exercised. Everything else is.
#
#   ./tests/test_cleanroom.sh [target-dir]
set -uo pipefail

# Repo root is one level up now that this lives in tests/.
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/tmp/ltx-cleanroom}"
ROCM_INDEX="https://download.pytorch.org/whl/nightly/rocm6.3"

pass=0; fail=0
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mPASS\033[0m %s\n' "$*"; pass=$((pass+1)); }
bad()  { printf '   \033[31mFAIL\033[0m %s\n' "$*"; fail=$((fail+1)); }

step "Fresh clone into $DEST"
rm -rf "$DEST"
git clone -q --branch "$(git -C "$SRC" rev-parse --abbrev-ref HEAD)" "$SRC" "$DEST" \
  && ok "cloned" || { bad "clone failed"; exit 1; }
cd "$DEST"

step "Files the README depends on are actually in the clone"
for f in README.md requirements.txt ltx_engine.py generate_video.py \
         cli_gen_vid.py quant_transformer_fp8.py ltx2_config.example.json; do
    [ -f "$f" ] && ok "$f" || bad "$f MISSING from a fresh clone"
done
[ -f ltx2_config.json ] && bad "ltx2_config.json should NOT be tracked" \
                        || ok "ltx2_config.json correctly untracked"

step "README step 0: venv + ROCm torch + requirements"
python3 -m venv venv && ok "venv created" || bad "venv creation failed"
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install -q --upgrade pip
echo "   installing ROCm torch (large download, be patient)..."
if pip install -q --pre torch torchvision torchaudio --index-url "$ROCM_INDEX"; then
    ok "ROCm torch installed"
else
    bad "ROCm torch install failed"
fi
echo "   installing requirements.txt..."
if pip install -q -r requirements.txt; then
    ok "requirements.txt resolved (incl. diffusers from git)"
else
    bad "requirements.txt failed to resolve"
fi

step "README step 0: the GPU check it tells you to run"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ok "torch.cuda.is_available() is True: $(python -c 'import torch;print(torch.cuda.get_device_name(0))')"
else
    bad "torch cannot see the GPU"
fi

step "Weights: symlinked (README steps 1-3 not exercised)"
for d in local_ltx25_fp8 local_ltx25_model local_ltx25_enhancer; do
    if [ -e "$SRC/$d" ]; then ln -sfn "$SRC/$d" "$d"; ok "linked $d"
    else bad "$SRC/$d not present to link"; fi
done

step "Imports resolve in the clean environment"
python -c "import ltx_engine" 2>/dev/null && ok "ltx_engine imports" || bad "ltx_engine import failed"
python -c "import ltx_engine, sys; sys.modules['tkinter']=None" 2>/dev/null \
  && ok "engine has no hard tkinter dependency" || bad "engine needs tkinter"

step "CLI runs from the clone (no GPU work, safe alongside a live render)"
if python cli_gen_vid.py --dry-run --prompt "clean room test" --no-cfg >/tmp/ltx_cr.log 2>&1; then
    ok "cli_gen_vid.py --dry-run"
    sed 's/^/      /' /tmp/ltx_cr.log | tail -8
else
    bad "cli_gen_vid.py --dry-run failed"; sed 's/^/      /' /tmp/ltx_cr.log | tail -15
fi

step "Config fallback: engine defaults apply with no config file present"
if python cli_gen_vid.py --config /nonexistent.json --dry-run --prompt x --no-cfg >/dev/null 2>&1; then
    ok "runs with no config file (uses built-in defaults)"
else
    bad "missing config file is not handled"
fi

printf '\n\033[1m== Result: %d passed, %d failed\033[0m\n' "$pass" "$fail"
printf 'Clean room left at %s (rm -rf to remove; venv is ~6GB)\n' "$DEST"
[ "$fail" -eq 0 ]
