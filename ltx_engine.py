#!/usr/bin/env python3
"""LTX-2.5 generation engine: everything that actually makes a video.

No UI of any kind lives here. Two front-ends drive it:
  generate_video.py  -- Tk control panel
  cli_gen_vid.py     -- headless CLI

`generation_worker()` marshals progress through objects passed in (anything
with .after/.set/.config), which is why the CLI can hand it plain stubs.
"""
__version__ = "0.1.1"

import os
import gc
import sys
import time
import json
import glob
import hashlib
import tempfile
import multiprocessing as mp


# =========================================================================
# 1. Environment Configuration
# =========================================================================
# VRAM is still the binding constraint (16GB), so allocator tuning stays.
# `expandable_segments` is the modern ROCm/CUDA fix for the fragmentation that
# `max_split_size_mb` used to paper over; it copes far better with the large,
# variable-sized tensors that VAE tiling and group-offload produce.
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.9",
)
os.environ["MIOPEN_LOG_LEVEL"] = "3"
# MIOPEN_FIND_MODE is deliberately NOT set. It used to be forced to "1"
# (NORMAL = exhaustive kernel search), which cost 13.8x on the VAE decode:
# measured 342.5s with it vs 24.8s without, same 1536x1024x49 shape, same
# 512/24 tiles, identical 6.63GB peak -- i.e. provably the same computation.
# The cost recurs on every run: exhaustive mode re-searches even when MIOpen's
# perf database is already populated, which is why decode was slow on repeat
# runs and not just the first. Leaving it unset uses MIOpen's own heuristic
# (dynamic hybrid). Export it yourself if you ever need to force a mode.
os.environ["AMD_DIRECT_DISPATCH"] = "1"
os.environ["HIP_FORCE_DEV_KERN_LAZY_COMPILE"] = "0"
os.environ["TORCH_ROCM_AOTXN_ENABLE"] = "1"
os.environ["AMD_SERIALIZE_KERNEL"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
# 16 threads: let the CPU-side maths (text encode, pinning,
# numpy postprocess, ffmpeg staging) actually use the box.
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

CONFIG_FILE = "ltx2_config.json"
# Local working state, same reasoning as ltx2_config.json: not shareable,
# not tracked.
PROMPT_HISTORY_FILE = "prompt_history.json"
PROMPT_HISTORY_MAX = 50
MODEL_PATH = os.path.abspath("./local_ltx25_fp8")
# LTX-2.5 model dir: holds the *correct* latent_upsampler config. Don't be
# tempted by Lightricks/ltxv-spatial-upscaler-0.9.7 -- that is an LTX-1 / 0.9.x
# upsampler and is architecturally incompatible with the LTX-2.5 VAE.
BASE_MODEL_PATH = os.path.abspath("./local_ltx25_model")
EMBED_CACHE_DIR = os.path.abspath("./.embed_cache")
# The LTX-2.5 checkpoints ship `prompt_enhancer` and `processor` as nulls,
# so the enhancer is a separate download -- Gemma-4, in two sizes. A keyed
# dict rather than one constant so the GUI can list/select/download by name
# without a new global here every time a size is added.
ENHANCER_MODELS = {
    "e2b": {
        "repo_id": "google/gemma-4-E2B-it",
        "local_dir": os.path.abspath("./local_ltx25_enhancer"),
        "label": "E2B (faster, ~9.6GB)",
    },
    "e4b": {
        "repo_id": "google/gemma-4-E4B-it",
        "local_dir": os.path.abspath("./local_ltx25_enhancer_e4b"),
        "label": "E4B (more accurate, ~15GB)",
    },
}
DEFAULT_ENHANCER_MODEL = "e4b"


def enhancer_is_downloaded(model_key):
    """config.json + at least one .safetensors file, not just directory
    existence -- an interrupted/partial download would otherwise read as
    "downloaded" here and then fail deep inside
    from_pretrained(local_files_only=True) instead of prompting a re-fetch."""
    local_dir = ENHANCER_MODELS[model_key]["local_dir"]
    if not os.path.isfile(os.path.join(local_dir, "config.json")):
        return False
    return any(f.endswith(".safetensors") for f in os.listdir(local_dir))


def download_enhancer_model(model_key):
    """Subprocess target: fetch a prompt-enhancer checkpoint from the Hub.
    huggingface_hub prints its own tqdm progress bars, which
    run_subprocess_logged streams into the GUI log the same way it does for
    every other long-running step in this file -- no separate progress
    plumbing needed."""
    from huggingface_hub import snapshot_download
    info = ENHANCER_MODELS[model_key]
    snapshot_download(info["repo_id"], local_dir=info["local_dir"])

# Auto Duration's duration_head can predict up to 20s -- that's a property of
# the head itself, not the hardware. What the *card* survives depends on
# resolution and VRAM, so the actual cap is computed per-run below rather than
# fixed at whatever was safe on one 16GB card at one resolution.
AUTO_DURATION_HARD_CEILING_S = 20.0

# The stock negative prompt, restored to upstream's. diffusers'
# DEFAULT_NEGATIVE_PROMPT drops five leading tokens that Lightricks' own
# constants.py carries (ltx-pipelines/src/ltx_pipelines/utils/constants.py):
# has_subtitles, has_blurbox, transition from black, transition to black,
# speech_ending_short. Checked 2026-08-26: those five are the only difference,
# and they are the functionally distinctive ones -- they read as training
# caption tags, so negating them suppresses burned-in subtitles, letterbox
# bars, fade-from-black openings and clipped speech endings. Everything else
# in the string is generic "blurry, low contrast" filler that any negative
# prompt would carry. Only has an effect in CFG mode; the distilled schedule
# is guidance-free and never evaluates the negative branch.
def default_negative_prompt():
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
    missing = ("has_subtitles, has_blurbox, transition from black, "
               "transition to black, speech_ending_short, ")
    if DEFAULT_NEGATIVE_PROMPT.startswith(missing):
        return DEFAULT_NEGATIVE_PROMPT
    return missing + DEFAULT_NEGATIVE_PROMPT

# Enabled, but read this before trusting `video_strength`: it is NOT a
# denoise/restyle dial, and its useful range is narrow for a specific,
# understood reason. Traced 2026-08-26 through diffusers + upstream ltx-core.
#
# WHY THE RANGE IS NARROW. We pass the source clip as an LTX2VideoCondition at
# `index=0`. In diffusers that routes to apply_first_frame_conditioning(),
# i.e. upstream's `VideoConditionByLatentIndex` -- which *overwrites* canvas
# tokens and sets `denoise_mask = 1 - strength` over them. Because the
# condition spans the whole clip, every token gets the same mask (verified:
# per-frame correlation to the source is flat across all 49 frames, slope
# -0.00016/frame -- frame 0 is not privileged). The denoise loop then does
#     x0 = model_out * (1 - mask) + clean_source * mask
# on EVERY step. That per-step pull compounds: the model's free contribution
# decays roughly as (1 - strength)**num_steps, so on the 8-step distilled
# schedule s=0.2 leaves ~17% and s=0.4 ~2%. Measured sweep at fixed seed
# (256x256/49f, corr between outputs): s=0.4 vs s=1.0 = 0.984, s=0.6 vs
# s=1.0 = 0.994, s=0.8 vs s=1.0 = 0.997, while s=0.0 vs any s>=0.2 is ~0.16.
# So it is a near-step function, not a ramp -- everything from ~0.2 up is the
# source. Corollary worth knowing: the usable range is a function of STEP
# COUNT, so it gets narrower still on longer schedules.
#
# WHAT UPSTREAM ACTUALLY USES FOR V2V. Per Lightricks' own docs
# (packages/ltx-pipelines/docs/conditioning.md: "Video Conditioning
# (ICLoraPipeline only) ... Uses VideoConditionByKeyframeIndex"), real v2v
# does not touch the canvas at all -- the reference is *appended* to the
# token sequence as extra clean tokens the transformer attends to
# (ltx-core's VideoConditionByReferenceLatent), and the dial is
# `conditioning_attention_strength`, which scales cross-attention scores
# rather than blending pixels. That is why theirs has a usable mid-range and
# this does not. It needs an IC-LoRA trained to attend across ref<->target;
# in diffusers that is LTX2InContextPipeline + LTX2ReferenceCondition (see
# the LORA_LOADING_ENABLED note below -- the 2.3-22b IC-LoRAs are the ones
# LTX's own 2.5 ComfyUI workflows load, so they are version-compatible).
#
# So: what is here is a source/generation crossfade that happens to be usable
# near s~0.9-1.0 for small prompt-nudged edits. It is not restyling, and no
# amount of tuning this number makes it restyling.
VIDEO_TO_VIDEO_ENABLED = True

# Plain LoRA loading works (confirmed -- a Cinemagraph LoRA produced a real,
# visible effect: locked composition vs. the base model's normal camera
# drift). Most LoRAs published for LTX-2 are IC-LoRAs (in-context
# conditioning -- Union-Control, Relight, Day-To-Night, Water-Simulation,
# etc.), which need a reference video and LTX2InContextPipeline. That is
# still not built, so an IC-LoRA loaded through this plain path will apply
# its weights and then do nothing useful -- there is no reference for it to
# attend to.
#
# What was measured 2026-08-26 (against LTX-2.3-22b-IC-LoRA-Union-Control,
# rank 64, on this 2.5 fp8 stack), correcting earlier guesses here:
#   - It does NOT crash. `pipe.load_lora_weights()` accepts the file as-is:
#     diffusers' LTX2LoraLoaderMixin already converts Lightricks' key format
#     ("diffusion_model.*" -> "transformer.*"), all 960 tensors, no shim.
#   - A 2.3-22b IC-LoRA is structurally compatible with the 2.5 transformer:
#     all 960 targeted modules exist, blocks 0..47 line up exactly, and all
#     480 A/B pairs are shape-clean. (LTX's own 2.5 ComfyUI workflows load
#     the 2.3-22b adapters, which is consistent with this.)
#   - All 960 adapter tensors land in fp8, so the fp8 -> bf16 cast below is
#     required for IC-LoRAs too, not just plain ones.
#   - There is no universal trigger-phrase prompt format. It is per-adapter:
#     Day-To-Night takes a plain descriptive prompt with a trailing "only
#     the lighting changes..." clause; Ingredients uses a two-part
#     "Reference sheet: / Generated video:" form. It belongs in a per-adapter
#     registry, not a global rule.
#   - Reference cost is set by the adapter's own `reference_downscale_factor`
#     safetensors metadata key (Union-Control ships 2 = half-res reference,
#     so +25% tokens; Day-To-Night ships 1, so +100%). Read it, don't assume.
# So the remaining work for real v2v is the pipeline and the reference
# plumbing, not LoRA loading -- see VIDEO_TO_VIDEO_ENABLED above for why the
# current v2v mode is a crossfade rather than a restyle.
LORA_LOADING_ENABLED = True

# From the VAE config: spatial_compression_ratio 32, temporal_compression_ratio 8.
# Width/height must be multiples of 32; frame counts must satisfy 8k + 1.
SPATIAL_COMPRESSION = 32
TEMPORAL_COMPRESSION = 8
MIN_DIMENSION = 256

# --- VRAM model for the token warning -----------------------------------------
# Fitted to two steady-state points from one instrumented 145-frame 2-stage run
# (stage 1: 7,296 tokens -> 9.00GB; stage 2: 29,184 tokens -> 12.97GB):
#
#     VRAM_GB ~= 7.68 + 1.814e-4 * latent_tokens
#
# The intercept is everything that doesn't scale with sequence length: streamed
# transformer weights, VAE + audio VAE + vocoder, allocator slack, and the
# desktop compositor (0.54GB idle on this box).
#
# CAVEAT: fitted on a single RX 9070 XT from a single run, so treat it as a
# calibrated guess, not physics. Both numbers are overridable in
# ltx2_config.json, and `token_warn_threshold` overrides the result outright.
VRAM_BASE_GB = 7.68
FIT_DESKTOP_BASELINE_GB = 0.54  # desktop compositor's idle share on the fit machine
VRAM_GB_PER_TOKEN = 1.814e-4
VRAM_HEADROOM = 0.85          # leave 15% for spikes the steady state misses
TOKEN_WARN_FALLBACK = 30000   # if VRAM can't be read: the measured 16GB value

# Estimated, NOT yet measured on real hardware (no >=24GB card to test
# against -- see CLAUDE.md "measure before claiming" / "state what is
# untested"). Lets a card with enough headroom skip group offload entirely
# and keep the transformer GPU-resident instead of streaming it block-by-block
# from pinned host RAM, which should be meaningfully faster once PCIe
# round-trips aren't in the critical path. Built from: 18GB on-disk fp8
# transformer (1 byte/param, so VRAM footprint matches) + 6.63GB measured VAE
# tile-decode peak (see benchmarks/bench_vae_tiles.py) + ~3GB estimated
# headroom for CFG/STG's extra forward pass and prompt-embed/latent buffers +
# ~1GB desktop baseline, rounded up. Re-tune this once real hardware is
# available: compare it against the peak-VRAM figure generation_worker
# already logs at the end of a full-resident run.
FULL_RESIDENT_VRAM_THRESHOLD_GB = 29.0


def token_warn_threshold(config=None):
    """Latent-token count above which to warn, scaled to the card actually
    present. VRAM_BASE_GB's fixed intercept assumed ~0.54GB of desktop
    compositor usage (the one machine this was fitted on). `hw_monitor`
    samples actual VRAM use at import time -- before any model is ever loaded
    -- as DESKTOP_BASELINE_GB, so a heavier desktop at launch (more windows,
    more monitors) than the fit machine had subtracts extra headroom instead
    of being silently absorbed. It can't be resampled live during a session:
    the model stays VRAM-resident across generations (see _MODEL_CACHE), so a
    live reading after the first run would double-count the resident weights
    that VRAM_BASE_GB already accounts for. Falls back to the measured 16GB
    figure if VRAM is unreadable."""
    if config and config.get("token_warn_threshold"):
        return int(config["token_warn_threshold"])
    _, _, vram_total = hw_monitor.get_gpu_stats()
    if not vram_total:
        return TOKEN_WARN_FALLBACK
    extra_desktop_gb = max(0.0, hw_monitor.desktop_baseline_gb - FIT_DESKTOP_BASELINE_GB)
    tokens = (vram_total * VRAM_HEADROOM - VRAM_BASE_GB - extra_desktop_gb) / VRAM_GB_PER_TOKEN
    # A card too small to hold the base footprint gets the floor, not a
    # negative threshold -- it will warn on essentially everything, correctly.
    return max(2000, int(tokens))


def resolve_modality_scale(config):
    """Modality-guidance scale from a config, honouring the legacy key name.

    Lives here rather than in each front-end because the GUI, CLI and server
    all need it for their pre-flight VRAM estimate, and a front-end reading
    the old key while the engine reads the new one would under-count a pass.
    """
    return float(config.get("modality_scale",
                            config.get("cfg_modality_scale", 1.0)))


def auto_duration_cap_s(width, height, upscale, cfg_mode, stg_mode, fps,
                        modality_scale=1.0, config=None):
    """Longest clip Auto Duration may pick at this resolution/guidance mode
    without exceeding token_warn_threshold() on the card actually present --
    replaces a flat seconds figure that was only ever valid for one 16GB card
    at one resolution."""
    threshold = token_warn_threshold(config)
    passes = guidance_pass_count(cfg_mode, stg_mode, modality_scale)
    scale = 2 if upscale else 1
    tokens_per_frame = ((height * scale) // 32) * ((width * scale) // 32)
    if tokens_per_frame <= 0 or passes <= 0:
        return AUTO_DURATION_HARD_CEILING_S
    lat_f_max = max(1, int(threshold / passes / tokens_per_frame))
    frames_max = (lat_f_max - 1) * 8 + 1
    return min(AUTO_DURATION_HARD_CEILING_S, frames_max / float(fps))

# --- Process-lifetime caches -------------------------------------------------
# Peak RAM use is ~45GB, so on anything comfortably above that there is no
# reason to re-read 18GB of transformer weights
# (or 23GB of text encoder) from disk on every click of "Generate".
_MODEL_CACHE = {"pipe": None, "path": None, "opts": None}
_UPSAMPLER_CACHE = {"model": None}
# Bounded: each entry is a full set of prompt embeddings, ~385MB (770MB with a
# negative prompt encoded, e.g. under CFG mode). Editing the prompt a few times
# in one GUI session used to grow this without limit. The disk cache in
# EMBED_CACHE_DIR still backs everything evicted here, so dropping an entry
# only costs a torch.load on reuse, not a re-encode.
_EMBED_MEM_CACHE_MAX = 3
_EMBED_MEM_CACHE = {}

cancel_flag = False
# Toggled live by the Debug checkbox. Read on every dbg() call, so flipping it
# mid-run takes effect immediately -- no restart, no reloading 18GB of weights.
debug_flag = False

class CancellationError(Exception):
    pass


def set_diffusers_verbosity(verbose):
    """`diffusers.logging` is a lazily-bound submodule: if the worker thread is
    mid-`import diffusers` when this runs, the module object in sys.modules is
    still incomplete and the attribute lookup raises. Import the submodule
    directly and never let a logging tweak take down the caller."""
    try:
        from diffusers.utils import logging as diffusers_logging
        if verbose:
            diffusers_logging.set_verbosity_info()
        else:
            diffusers_logging.set_verbosity_error()
    except Exception as exc:
        print(f"  [dbg] could not set diffusers verbosity: {exc}")


def dbg(msg):
    """Debug line tagged with the numbers that actually matter here: VRAM is the
    binding constraint on 16GB, and RAM is what the offload path trades against."""
    if not debug_flag:
        return
    _, vram_used, vram_total = hw_monitor.get_gpu_stats()
    ram_used, ram_total = hw_monitor.get_ram_stats()
    vram = f"{vram_used:.2f}/{vram_total:.1f}GB" if vram_used is not None else "n/a"
    print(f"  [dbg] {msg}  (VRAM {vram}, RAM {ram_used:.1f}/{ram_total:.0f}GB)")

# =========================================================================
# 2. Hardware Monitoring 
# =========================================================================
class LinuxHardwareMonitor:
    def __init__(self):
        self.sysfs_gpu_path = None
        self.last_cpu_total = None
        self.last_cpu_cores = {}
        
        max_vram = 0
        for i in range(10):
            path = f"/sys/class/drm/card{i}/device"
            vram_path = os.path.join(path, "mem_info_vram_total")
            if os.path.exists(vram_path):
                try:
                    with open(vram_path, "r") as f:
                        vram = int(f.read().strip())
                    if vram > max_vram:
                        max_vram = vram
                        self.sysfs_gpu_path = path
                except Exception:
                    pass
        self.get_cpu_stats()
        # Sampled here, before any model is ever loaded, so it's whatever the
        # desktop itself holds -- compositor, open windows, monitor count --
        # not the pipeline's own footprint. See token_warn_threshold().
        _, desktop_used, _ = self.get_gpu_stats()
        self.desktop_baseline_gb = desktop_used or 0.0

    def get_cpu_stats(self):
        try:
            with open("/proc/stat", "r") as f:
                lines = f.readlines()
            core_pcts, avg_pct = [], 0.0
            for line in lines:
                parts = line.split()
                if not parts: continue
                name = parts[0]
                if name == "cpu":
                    times = [float(x) for x in parts[1:8]]
                    idle, total = times[3] + times[4], sum(times)
                    if self.last_cpu_total is not None:
                        diff_idle = idle - self.last_cpu_total[0]
                        diff_total = total - self.last_cpu_total[1]
                        if diff_total > 0:
                            avg_pct = max(0.0, min(100.0, (1.0 - (diff_idle / diff_total)) * 100.0))
                    self.last_cpu_total = (idle, total)
                elif name.startswith("cpu") and name[3:].isdigit():
                    idx = int(name[3:])
                    times = [float(x) for x in parts[1:8]]
                    idle, total = times[3] + times[4], sum(times)
                    pct = 0.0
                    if idx in self.last_cpu_cores:
                        diff_idle = idle - self.last_cpu_cores[idx][0]
                        diff_total = total - self.last_cpu_cores[idx][1]
                        if diff_total > 0:
                            pct = max(0.0, min(100.0, (1.0 - (diff_idle / diff_total)) * 100.0))
                    self.last_cpu_cores[idx] = (idle, total)
                    core_pcts.append(pct)
            return avg_pct, core_pcts
        except Exception:
            return 0.0, []

    def get_ram_stats(self):
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            mem_data = {p.split(":")[0].strip(): int(p.split(":")[1].strip().split()[0]) for p in lines if ":" in p}
            total_kb = mem_data.get("MemTotal", 0)
            avail_kb = mem_data.get("MemAvailable", mem_data.get("MemFree", 0))
            return (total_kb - avail_kb) / (1024**2), total_kb / (1024**2)
        except Exception:
            return 0.0, 0.0

    def get_gpu_stats(self):
        if not self.sysfs_gpu_path: return None, None, None
        try:
            with open(os.path.join(self.sysfs_gpu_path, "gpu_busy_percent"), "r") as f:
                gpu_usage = int(f.read().strip())
            with open(os.path.join(self.sysfs_gpu_path, "mem_info_vram_used"), "r") as f:
                vram_used = int(f.read().strip()) / (1024**3)
            with open(os.path.join(self.sysfs_gpu_path, "mem_info_vram_total"), "r") as f:
                vram_total = int(f.read().strip()) / (1024**3)
            return gpu_usage, vram_used, vram_total
        except Exception:
            return None, None, None

hw_monitor = LinuxHardwareMonitor()



# =========================================================================
# 3. Subprocess plumbing + Stage 1 Text Encoding
# =========================================================================
def _subprocess_entry(log_path, target_name, args):
    """Child-side entry point. Redirects fd 1/2 to a file *before* running the
    real target, so the parent can stream it into the GUI log.

    Redirecting at the file-descriptor level (not just sys.stdout) is what
    catches tqdm bars, C-level library chatter and, most importantly, the
    traceback of a child that dies -- which otherwise vanishes into the
    terminal nobody is looking at.
    """
    import os as _os
    import sys as _sys
    fd = _os.open(log_path, _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC, 0o644)
    _os.dup2(fd, 1)
    _os.dup2(fd, 2)
    _sys.stdout = _os.fdopen(1, "w", buffering=1)   # line buffered
    _sys.stderr = _os.fdopen(2, "w", buffering=1)
    try:
        globals()[target_name](*args)
    finally:
        _sys.stdout.flush()
        _sys.stderr.flush()


def run_subprocess_logged(target_name, args, cancel_error=None):
    """Run `target_name` in a spawned process, streaming its output into the
    GUI log (this process's stdout) rather than the launching terminal.

    Returns the child's exit code. Raises `cancel_error` if cancel_flag is set
    while it runs.
    """
    fd, log_path = tempfile.mkstemp(prefix="ltx_subproc_", suffix=".log")
    os.close(fd)
    proc = mp.Process(target=_subprocess_entry, args=(log_path, target_name, args))
    proc.start()
    pos = 0
    try:
        while True:
            alive = proc.is_alive()
            try:
                with open(log_path, "r", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                if chunk:
                    print(chunk, end="")
            except FileNotFoundError:
                pass
            if not alive:
                break
            if cancel_error is not None and cancel_flag:
                proc.terminate()
                proc.join()
                raise cancel_error
            proc.join(timeout=0.25)
        return proc.exitcode
    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass


def snap_dimension(v):
    """Round a width/height to what the VAE can encode (32:1 spatial)."""
    return max(MIN_DIMENSION, round(v / SPATIAL_COMPRESSION) * SPATIAL_COMPRESSION)


def align_frames(n):
    """Snap a frame count to the 8k+1 rule (VAE is 8:1 temporal)."""
    return (max(1, round((n - 1) / 8)) * 8) + 1


def load_prompt_history():
    """Most-recent-first. Missing/corrupt file reads as empty -- history is a
    convenience, not something worth failing a run over."""
    try:
        with open(PROMPT_HISTORY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def record_prompt_history(prompt, negative_prompt, config):
    """Called once per attempted generation (see generation_worker) -- logs
    what was actually run, not every keystroke. Deduplicates consecutive
    identical prompts (re-running the same one shouldn't spam the list) and
    caps at PROMPT_HISTORY_MAX, dropping the oldest."""
    if not prompt.strip():
        return
    history = load_prompt_history()
    if history and history[0].get("prompt") == prompt:
        return
    history.insert(0, {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "seed": config.get("active_seed"),
        "width": config.get("width"),
        "height": config.get("height"),
    })
    try:
        with open(PROMPT_HISTORY_FILE, "w") as f:
            json.dump(history[:PROMPT_HISTORY_MAX], f, indent=2)
    except OSError as exc:
        print(f"  [!] Could not save prompt history: {exc}")


def clear_prompt_history():
    """Delete the history file outright rather than writing `[]` -- avoids
    leaving an empty-but-present file that would need distinguishing from
    "no history yet" everywhere load_prompt_history() is used."""
    try:
        os.remove(PROMPT_HISTORY_FILE)
    except FileNotFoundError:
        pass


def latent_tokens(width, height, frames, upscale):
    """Transformer sequence length of the final stage -- the number the VRAM
    warning is built on. Shared by the GUI and the CLI so they can't disagree."""
    scale = 2 if upscale else 1
    lat_f = (frames - 1) // 8 + 1
    return lat_f * ((height * scale) // 32) * ((width * scale) // 32)


def guidance_pass_count(cfg_mode, stg_mode, modality_scale=1.0):
    """How many full transformer forward passes stage 1 runs per step.

    Base is 1. CFG and STG each add one (see the comments in
    generation_worker for why -- CFG batches cond+uncond into one doubled
    call, STG and modality guidance are each their own separate call).
    Shared by the pre-flight VRAM estimate in both front-ends and the engine's
    own debug line, so a config that trips STG or modality guidance can't be
    under-estimated in one place and correctly estimated in another --
    exactly the drift the "eff_tokens = tokens*2 if cfg else tokens" shortcut
    used to have: STG alone, or CFG+STG, were both estimated as if they cost
    what plain CFG costs.
    """
    return 1 + (1 if cfg_mode else 0) + (1 if stg_mode else 0) + \
        (1 if modality_scale > 1.0 else 0)


def _embed_cache_key(model_path, p, np_text, need_negative):
    h = hashlib.sha256()
    # The negative text only belongs in the key when it is actually encoded.
    # Under the distilled (guidance-free) schedule need_negative is False and
    # `encode_prompt` is handed None, so including the typed text here would
    # force a full re-encode that returns byte-identical embeddings.
    parts = (model_path, p, (np_text or "") if need_negative else "",
             "neg" if need_negative else "noneg")
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# diffusers ships I2V/T2V system prompts (LTX2_5_I2V/T2V_DEFAULT_SYSTEM_PROMPT)
# but no FLF2V one -- FLF2V is a diffusers pipeline capability
# (LTX2ConditionPipeline), not something LTX2Pipeline.enhance_prompt() itself
# knows about (it hard-codes a single `{"type": "image"}` content block, see
# _enhance_flf2v_inproc below). Adapted from LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT:
# same 7-point captioning style, but grounds BOTH ends of the clip and asks
# for one continuous, physically plausible path between them instead of one
# open-ended continuation.
LTX2_5_FLF2V_SYSTEM_PROMPT = """You are given a REFERENCE START IMAGE (the exact first frame of the video), a REFERENCE END IMAGE (the exact last frame of the video), and a user's short request describing what happens between them. Write a single, highly detailed audio-visual caption describing the video that begins on the start image, ends on the end image, and best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

FIRST+LAST FRAME GROUNDING (do this first): the opening of your caption must match the START image exactly -- same subject(s), identity, appearance, clothing, setting, lighting, and composition as shown. The closing of your caption must match the END image exactly, in the same way. Narrate one continuous, physically plausible chronological path from the start image to the end image -- use the user's request to determine what happens in between, and never contradict, replace, or invent anything inconsistent with either image. Single continuous take -- no hard cuts.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements — grounded in and consistent with BOTH reference images at their respective ends of the clip. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up) — consistent with how the start image is framed, and if the end image implies a different framing, describe the camera move that gets there.
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best bridges the two reference frames.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle) — matching the start image's viewpoint at the opening, and the end image's viewpoint by the close.
   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep the user's requested motion/action central and in motion throughout, arriving at the end image's exact state by the final moment.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style or contradicting either reference image): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene, moving from the start frame to the end frame, LOOKS at its most visually striking, never adding new objects or actions and never contradicting either reference image. Keep everything else (first+last frame grounding, framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified."""


def _enhance_flf2v_inproc(torch, pipe_text, prompt, start_image, end_image, system_prompt, device):
    """Two-image variant of `LTX2Pipeline.enhance_prompt` -- that method only
    ever inserts one `{"type": "image"}` content block (see
    diffusers/pipelines/ltx2/pipeline_ltx2.py), so FLF2V's start+end pair
    can't go through it. Replicates its message/generation/decode logic
    exactly, just with two image slots instead of one -- verified against
    this checkpoint that the Gemma4 processor accepts a list of 2 images
    matched positionally to 2 `{"type": "image"}` blocks in the template.
    """
    from diffusers.pipelines.ltx2.pipeline_ltx2 import _prepare_enhance_image, _pad_inputs_for_attention_alignment, clean_response
    from diffusers.pipelines.ltx2.utils import GEMMA4_PROMPT_ENHANCEMENT_CONFIG

    config = GEMMA4_PROMPT_ENHANCEMENT_CONFIG
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
        ]},
    ]
    template = pipe_text.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [_prepare_enhance_image(start_image), _prepare_enhance_image(end_image)]
    model_inputs = pipe_text.processor(text=template, images=images, return_tensors="pt").to(device)
    pad_token_id = pipe_text.processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)
    pipe_text.prompt_enhancer.to(device)
    torch.manual_seed(config.seed)
    generated = pipe_text.prompt_enhancer.generate(
        **model_inputs, max_new_tokens=config.max_new_tokens, **config.generation_kwargs,
    )
    generated_ids = generated[0][len(model_inputs.input_ids[0]):]
    return clean_response(pipe_text.processor.tokenizer.decode(generated_ids, skip_special_tokens=True))


def _enhance_prompt_inproc(torch, p, image_path, max_words=None, end_image_path=None, model_key=None):
    """Run the Gemma-4 prompt enhancer. Always called inside a throwaway
    subprocess so the ~10-16GB enhancer never coexists with the transformer.

    `LTX2Pipeline.enhance_prompt` only reads `.processor`, `.prompt_enhancer`
    and `._execution_device`, so it runs against a bare shell object -- no need
    to load the 23GB text encoder just to rewrite a sentence.
    """
    import types
    model_key = model_key or DEFAULT_ENHANCER_MODEL
    enhancer_info = ENHANCER_MODELS[model_key]
    enhancer_path = enhancer_info["local_dir"]
    from diffusers import LTX2Pipeline
    from transformers import AutoModelForCausalLM, AutoProcessor
    from diffusers.pipelines.ltx2.utils import (
        LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT,
        LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT,
    )

    image = None
    if image_path and os.path.exists(image_path):
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

    end_image = None
    if end_image_path and os.path.exists(end_image_path):
        from PIL import Image
        end_image = Image.open(end_image_path).convert("RGB")

    print(f"  -> Loading prompt enhancer (Gemma-4 {enhancer_info['label']})...")
    pipe_text = types.SimpleNamespace(_execution_device="cpu")
    pipe_text.enhance_prompt = LTX2Pipeline.enhance_prompt.__get__(pipe_text)
    pipe_text.processor = AutoProcessor.from_pretrained(enhancer_path, local_files_only=True)
    pipe_text.prompt_enhancer = AutoModelForCausalLM.from_pretrained(
        enhancer_path, dtype=torch.bfloat16, local_files_only=True,
    ).eval()

    use_flf2v = image is not None and end_image is not None
    if use_flf2v:
        system_prompt = LTX2_5_FLF2V_SYSTEM_PROMPT
    elif image is not None:
        system_prompt = LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT
    else:
        system_prompt = LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT

    # The stock system prompt asks for an "exhaustive and lossless" caption --
    # that verbosity is deliberate, because LTX-2.5 was trained on captions in
    # that style. Capping `max_new_tokens` instead would just truncate a
    # sentence mid-clause, so constrain it in the instructions and let the model
    # decide what to drop. Appended rather than replacing, to keep the trained
    # style intact.
    if max_words:
        system_prompt += (
            f"\n\nLENGTH LIMIT: the caption must be no more than {max_words} words. "
            "Stay within it by cutting background scenery, secondary objects and "
            "incidental ambience -- never the subject, its action, the camera "
            "movement, or anything the user explicitly asked for. Prefer one "
            "precise adjective to three vague ones. Still one continuous "
            "paragraph, same style."
        )

    def run(device):
        with torch.no_grad():
            if use_flf2v:
                return _enhance_flf2v_inproc(torch, pipe_text, p, image, end_image, system_prompt, device)
            return pipe_text.enhance_prompt(
                prompt=p, system_prompt=system_prompt, image=image, device=device,
            )[0]

    try:
        enhanced = run("cuda")
    except torch.OutOfMemoryError:
        print("  -> Enhancer OOM on GPU; retrying on CPU (minutes, not seconds).")
        pipe_text.prompt_enhancer.to("cpu")
        torch.cuda.empty_cache()
        enhanced = run("cpu")

    # Free the enhancer before the text encoder runs; both in RAM at once is ~33GB.
    pipe_text.prompt_enhancer = None
    pipe_text.processor = None
    gc.collect()
    torch.cuda.empty_cache()

    # NB: not printed here -- the parent prints it into the GUI log after
    # reading the result file, so it appears next to the prompt box.
    return enhanced


def enhance_in_subprocess(p, image_path, out_path, max_words=None, end_image_path=None, model_key=None):
    """Subprocess target for the standalone '✨ Enhance Now' button."""
    import torch
    with open(out_path, "w") as f:
        f.write(_enhance_prompt_inproc(torch, p, image_path, max_words, end_image_path, model_key))


def encode_in_subprocess(model_path, p, np_text, need_negative, out_path):
    import torch
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    from diffusers import LTX2Pipeline
    print("  -> Booting Text Encoder in isolated process...")
    pipe_text = LTX2Pipeline.from_pretrained(
        model_path,
        transformer=None, vae=None, audio_vae=None, vocoder=None,
        torch_dtype=torch.bfloat16, local_files_only=True,
    )
    with torch.no_grad():
        # When CFG is disabled (distilled schedule) the negative branch is never
        # evaluated, so skip encoding it entirely -- that halves this stage.
        embeds = pipe_text.encode_prompt(
            prompt=p,
            negative_prompt=np_text if need_negative else None,
            do_classifier_free_guidance=need_negative,
            device="cpu",
        )
    # Write to a temp file and rename into place. A straight save to out_path
    # left a truncated .pt behind if the process was killed mid-write --
    # cancel does exactly that (run_subprocess_logged terminates on
    # cancel_flag) -- and every later run for that same prompt would find the
    # file present, load it, and fail with an unpickling error nowhere near
    # the actual cause. os.replace is atomic on the same filesystem, so
    # readers only ever see a complete file or none at all.
    tmp_path = out_path + f".tmp{os.getpid()}"
    torch.save(embeds, tmp_path)
    os.replace(tmp_path, out_path)
    print("  -> Encoding complete. Terminating process to release RAM.")

# =========================================================================
# 4. Background Generation Thread
# =========================================================================
def _build_latent_upsampler(torch, local_files_only=True):
    """
    Load the LTX-2.5 spatial x2 latent upsampler.

    NOTE: the standalone Lightricks/ltxv-spatial-upscaler-0.9.7 release is an
    *LTX-1 / 0.9.x* upsampler (`LTXLatentUpsamplerModel` + `AutoencoderKLLTXVideo`)
    and is architecturally incompatible with the LTX-2.5 VAE -- don't wire it in
    here. The real LTX-2.5 weights live in
    ./local_ltx25_fp8/latent_upscale_models/*spatial-upscaler*.safetensors and the
    matching config in ./local_ltx25_model/latent_upsampler/config.json.
    """
    import json as _json
    from safetensors.torch import load_file
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    if _UPSAMPLER_CACHE["model"] is not None:
        return _UPSAMPLER_CACHE["model"]

    model = None
    # Preferred: a proper diffusers subfolder with weights next to the config.
    for root_dir in (BASE_MODEL_PATH, MODEL_PATH):
        sub = os.path.join(root_dir, "latent_upsampler")
        if glob.glob(os.path.join(sub, "*.safetensors")) or glob.glob(os.path.join(sub, "*.bin")):
            model = LTX2LatentUpsamplerModel.from_pretrained(
                root_dir, subfolder="latent_upsampler",
                torch_dtype=torch.bfloat16, local_files_only=local_files_only,
            )
            break

    if model is None:
        # Fallback: raw checkpoint + standalone config (this is the layout we have).
        cfg_path = None
        for root_dir in (BASE_MODEL_PATH, MODEL_PATH):
            candidate = os.path.join(root_dir, "latent_upsampler", "config.json")
            if os.path.exists(candidate):
                cfg_path = candidate
                break

        weights = []
        for root_dir in (MODEL_PATH, BASE_MODEL_PATH):
            weights += sorted(glob.glob(os.path.join(root_dir, "latent_upscale_models", "*spatial*.safetensors")))
        if not weights:
            raise FileNotFoundError(
                "Could not find an LTX-2.5 spatial latent upscaler .safetensors under "
                f"{MODEL_PATH}/latent_upscale_models or {BASE_MODEL_PATH}/latent_upscale_models."
            )
        weight_path = weights[0]

        if cfg_path is not None:
            cfg = {k: v for k, v in _json.load(open(cfg_path)).items() if not k.startswith("_")}
        else:
            # Every LTX-2.5 x2 spatial upscaler ships these dims.
            cfg = dict(in_channels=128, mid_channels=1024, num_blocks_per_stage=4, dims=3,
                       spatial_upsample=True, temporal_upsample=False,
                       rational_spatial_scale=2.0, use_rational_resampler=False)

        model = LTX2LatentUpsamplerModel(**cfg)
        state = load_file(weight_path)
        model.load_state_dict(state, strict=True)
        model = model.to(torch.bfloat16)
        print(f"  -> Latent upsampler loaded from {os.path.basename(weight_path)}")

    model.eval()
    _UPSAMPLER_CACHE["model"] = model
    return model


def generation_worker(config, root, progress_var, progress_bar, btn_generate, btn_cancel):
    pipe = None
    upscale_pipe = None
    try:
        print("\n[*] Initializing PyTorch and ROCm backends...")
        script_start_time = time.time()

        record_prompt_history(config["prompt"], config.get("negative_prompt", ""), config)

        import warnings
        import torch
        import torch.nn.functional as F
        import diffusers
        from diffusers import (
            LTX2Pipeline,
            LTX2VideoTransformer3DModel,
            LTX2LatentUpsamplePipeline,
            LTX2ImageToVideoPipeline,
            LTX2ConditionPipeline,
        )
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.hooks import apply_group_offloading
        # KNOWN DIVERGENCE FROM THE REFERENCE RECIPE (audited 2026-08-26).
        # These sigma values match upstream exactly, but the *sampler* stepping
        # through them does not. Lightricks' DistilledPipeline switches stage 1
        # to an ancestral (SDE) Euler step for checkpoints at 2.5 or newer --
        # `ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)` in
        # ltx-pipelines/src/ltx_pipelines/distilled.py, with eta=1.0,
        # s_noise=1.0; stage 2 stays deterministic because its 3-step tail is
        # too short to remove freshly injected noise. We run deterministic
        # Euler for both stages, because that is what diffusers does here and
        # what this checkpoint's own scheduler_config.json asks for
        # ("stochastic_sampling": false).
        #
        # Not fixable by flipping that flag, for two separate reasons:
        #   1. Different update rule. diffusers' stochastic branch is
        #      `prev = (1 - s_next) * x0 + s_next * noise` -- it discards the
        #      current sample entirely and re-draws from the forward process.
        #      Upstream's eta=1 steps to `sigma_down = s_next**2 / s` (0.848
        #      where diffusers would use 0.909) keeping a share of x, then
        #      renoises variance-preservingly. They are not the same step.
        #   2. It would break seed reproducibility. No LTX-2 pipeline in
        #      diffusers passes `generator=` to `scheduler.step()`, so the
        #      per-step noise would come from the global RNG -- and controlled
        #      same-seed A/Bs are what this project measures with.
        # Left as-is deliberately. Untested whether the ancestral sampler
        # actually improves output here; it is a recipe difference, not a
        # known defect.
        from diffusers.pipelines.ltx2.utils import (
            DISTILLED_SIGMA_VALUES,
            STAGE_2_DISTILLED_SIGMA_VALUES,
            LTX2_5_IMAGE_CRF,
        )
        from diffusers.utils import encode_video

        warnings.filterwarnings("ignore", category=FutureWarning)
        # Don't re-mute diffusers if the Debug checkbox is on.
        set_diffusers_verbosity(debug_flag)
        dbg(f"torch {torch.__version__}, diffusers {diffusers.__version__}, "
            f"backend={config.get('attention_backend', 'native')}")

        # FP8 Dynamic Patching (weights stay fp8 in RAM *and* on the PCIe wire;
        # only the active block is upcast to bf16 on-GPU inside the linear).
        fp8_types = {torch.float8_e4m3fn}
        if hasattr(torch, "float8_e4m3fnuz"):
            fp8_types.add(torch.float8_e4m3fnuz)

        def dynamic_fp8_linear_forward(self, input):
            weight = self.weight
            if weight.dtype in fp8_types:
                weight = weight.to(input.dtype)
            bias = self.bias.to(input.dtype) if self.bias is not None else None
            return F.linear(input, weight, bias)

        # NOTE: this patches the *class*, so it applies process-wide to every
        # torch.nn.Linear -- not just the transformer's, also the VAE's, audio
        # VAE's, vocoder's, connectors' and duration head's. Harmless today
        # (the branch is a dtype check that's a no-op for bf16 weights), but
        # it's a global mutation with no way back inside a long-lived worker
        # thread, and it silently forecloses relying on the stock fast path
        # for any Linear anywhere in the process from this point on.
        torch.nn.Linear.forward = dynamic_fp8_linear_forward

        def patch_transformer_fp8_params(module, target_dtype=torch.bfloat16):
            if isinstance(module, torch.nn.Linear):
                return
            for _name, param in module.named_parameters(recurse=False):
                if param.dtype in fp8_types:
                    param.data = param.data.to(target_dtype)
            for _child_name, child_module in module.named_children():
                patch_transformer_fp8_params(child_module, target_dtype)

        try:
            torch.set_num_threads(16)
            torch.set_num_interop_threads(16)
        except RuntimeError:
            pass

        use_upscale = bool(config.get("upscale"))
        use_image = config.get("mode") == "image2video"
        # FLF2V ("First-Last-Frame to Video") -- LTX's own name for start+end
        # frame conditioning, per the diffusers example's asset filenames
        # (flf2v_input_first_frame.png / flf2v_input_last_frame.png). A
        # separate mode from plain i2v, not an optional extra on it, since it
        # needs a different pipeline (LTX2ConditionPipeline, not
        # LTX2ImageToVideoPipeline) and a mandatory second image.
        use_flf2v = config.get("mode") == "flf2v"
        use_video = config.get("mode") == "video2video"
        if use_video and not VIDEO_TO_VIDEO_ENABLED:
            raise ValueError("Video-to-video is disabled (see VIDEO_TO_VIDEO_ENABLED "
                             "in ltx_engine.py).")

        input_image = None
        input_end_image = None
        if use_image or use_flf2v:
            from PIL import Image

            image_path = config.get("image_path", "")
            if not image_path or not os.path.exists(image_path):
                mode_desc = "First+Last-Frame" if use_flf2v else "Image-to-video"
                raise ValueError(f"{mode_desc} mode selected but no valid start image path was given: '{image_path}'")
            input_image = Image.open(image_path).convert("RGB")

            if use_flf2v:
                end_image_path = config.get("end_image_path", "")
                if not end_image_path or not os.path.exists(end_image_path):
                    raise ValueError(f"First+Last-Frame mode selected but no valid end image path was given: '{end_image_path}'")
                input_end_image = Image.open(end_image_path).convert("RGB")

        input_video_frames = None
        if use_video:
            from diffusers.utils import load_video

            video_path = config.get("video_path", "")
            if not video_path or not os.path.exists(video_path):
                raise ValueError(f"Video-to-video mode selected but no valid video path was given: '{video_path}'")
            input_video_frames = load_video(video_path)

        # The distilled LTX-2.5 schedule is guidance-free. Leaving
        # `audio_guidance_scale` at its 7.0 default silently turns CFG back ON
        # (`do_classifier_free_guidance` ORs the video and audio scales), which
        # doubles the transformer batch => ~2x the time and ~2x the activation VRAM.
        distilled_guidance = dict(
            guidance_scale=1.0,
            audio_guidance_scale=1.0,
            stg_scale=0.0,
            audio_stg_scale=0.0,
            modality_scale=1.0,
            audio_modality_scale=1.0,
        )

        # --- CFG quality mode -------------------------------------------------
        # Classifier-free guidance runs the transformer twice per step (prompt +
        # negative) and pushes away from the negative prediction. That is what
        # makes the model honour secondary details instead of settling for a
        # generically plausible scene -- at ~2x VRAM and ~2x time per step, on
        # top of needing a full schedule instead of the 8 distilled steps.
        #
        # Applied to stage 1 only: stage 2 is a short refinement tail that does
        # not set content, so leaving it guidance-free halves the VRAM cost of
        # this mode without meaningfully affecting adherence.
        use_cfg = bool(config.get("cfg_mode"))
        cfg_steps = max(8, int(config.get("cfg_steps", 30)))
        cfg_scale = float(config.get("cfg_scale", 3.0))
        # Audio CFG is a separate, much stronger knob in the reference recipe --
        # LTX_2_PARAMS.audio_guider_params.cfg_scale is 7.0 against 3.0 for
        # video (packages/ltx-pipelines/.../utils/constants.py:51,61), and
        # diffusers' own __call__ default for audio_guidance_scale is 7.0 too.
        # This used to reuse cfg_scale for both, running audio guidance at
        # less than half the reference strength. Costs nothing extra: audio
        # CFG rides the same doubled forward pass as video CFG.
        audio_cfg_scale = float(config.get("audio_cfg_scale", 7.0))

        # --- Spatio-Temporal Guidance ----------------------------------------
        # STG runs a second pass with one transformer block perturbed and pushes
        # away from that degraded prediction. Unlike CFG it uses NO negative
        # prompt -- it steers on structure, not text -- so it targets duplicated
        # limbs and objects that float free of the scene rather than prompt
        # adherence. One extra pass, i.e. 2x, and it works with the 8-step
        # distilled schedule, making it ~4x cheaper than CFG mode.
        #
        # It does work with the distilled sigmas despite those being trained
        # guidance-free -- tried at stg_scale 1.0 and it visibly improved
        # anatomy and object coherence. One comparison, not a controlled
        # measurement. If output looks over-sharpened rather than better
        # formed, lower stg_scale before abandoning it.
        use_stg = bool(config.get("stg_mode"))
        stg_scale = float(config.get("stg_scale", 1.0))
        stg_blocks = config.get("stg_blocks") or [28]

        # Modality guidance is NOT tied to CFG: diffusers gates it purely on
        # `modality_scale > 1.0 or audio_modality_scale > 1.0`
        # (do_modality_isolation_guidance, pipeline_ltx2.py), the same shape as
        # STG's gate. So it can ride the 8-step distilled schedule for one extra
        # pass, exactly like STG -- it does not require CFG's 30 doubled steps.
        # `cfg_modality_scale` is the older key name from when this was only
        # reachable inside CFG mode; it is still honoured so existing configs
        # keep working, but `modality_scale` is the one to set now.
        modality_scale = float(config.get("modality_scale",
                                          config.get("cfg_modality_scale", 1.0)))
        use_modality = modality_scale > 1.0

        if use_cfg:
            stage1_guidance = dict(
                guidance_scale=cfg_scale,
                audio_guidance_scale=audio_cfg_scale,
                # STG on top of CFG would be a third pass per step. Allowed, but
                # it is the user asking for it explicitly.
                stg_scale=stg_scale if use_stg else 0.0,
                # NOT 0.0: diffusers does `audio_stg_scale = audio_stg_scale or
                # stg_scale` (pipeline_ltx2.py:1149), and `0.0 or X` is X in
                # Python -- so 0.0 here gets silently replaced by the video
                # stg_scale above whenever STG is on, applying full-strength
                # audio STG we never asked for. A tiny nonzero value survives
                # the `or` as itself and is numerically off (it only scales a
                # delta that gets added in, see :1505).
                audio_stg_scale=1e-8,
                modality_scale=modality_scale,
                audio_modality_scale=modality_scale,
            )
            stage1_schedule = dict(num_inference_steps=cfg_steps)
            need_negative = True        # the negative prompt is finally encoded and used
        elif use_stg:
            # Distilled schedule kept -- only the extra perturbed pass is added.
            stage1_guidance = dict(
                guidance_scale=1.0,     # CFG stays off: no negative prompt involved
                audio_guidance_scale=1.0,
                stg_scale=stg_scale,
                # See the comment on the CFG+STG branch above: 0.0 here gets
                # silently replaced by `stg_scale` via diffusers' `or`
                # fallback, applying audio STG we never asked for.
                audio_stg_scale=1e-8,
                modality_scale=modality_scale,
                audio_modality_scale=modality_scale,
            )
            stage1_schedule = dict(sigmas=DISTILLED_SIGMA_VALUES)
            need_negative = False
        else:
            stage1_guidance = dict(distilled_guidance)  # copy: stage 2 reuses
            # distilled_guidance directly (**distilled_guidance below), so an
            # in-place edit of stage1_guidance here would leak into stage 2
            stage1_guidance["modality_scale"] = modality_scale
            stage1_guidance["audio_modality_scale"] = modality_scale
            stage1_schedule = dict(sigmas=DISTILLED_SIGMA_VALUES)
            need_negative = False       # CFG off => negative branch is never evaluated

        if use_stg:
            stage1_guidance["spatio_temporal_guidance_blocks"] = stg_blocks

        stage1_w, stage1_h = int(config["width"]), int(config["height"])
        out_w, out_h = (stage1_w * 2, stage1_h * 2) if use_upscale else (stage1_w, stage1_h)

        # Auto Duration: with a `duration_head` present the model predicts clip
        # length from the prompt when `num_frames` is omitted. Left unbounded it
        # will happily pick up to 20s -- clamp to whatever this card actually
        # survives at this resolution/guidance mode (see auto_duration_cap_s()).
        want_auto_duration = bool(config.get("auto_duration"))
        auto_cap_s = auto_duration_cap_s(
            stage1_w, stage1_h, use_upscale, use_cfg, use_stg,
            float(config["fps"]), modality_scale, config)
        auto_min_s = float(config.get("auto_min_seconds", 2.0))
        auto_max_s = min(float(config.get("auto_max_seconds", 5.0)), auto_cap_s)
        auto_min_s = min(auto_min_s, auto_max_s - 0.1)

        total_steps = (cfg_steps if use_cfg else len(DISTILLED_SIGMA_VALUES)) \
            + (len(STAGE_2_DISTILLED_SIGMA_VALUES) if use_upscale else 0)
        root.after(0, lambda: progress_bar.config(maximum=total_steps))
        root.after(0, progress_var.set, 0)

        # --- Stage 1: Prompt embeddings (memory -> disk -> subprocess) ---
        print("\n--- [1/4] Resolving Prompt Embeddings ---")
        key = _embed_cache_key(MODEL_PATH, config["prompt"], config["negative_prompt"], need_negative)
        embeds = _EMBED_MEM_CACHE.get(key)

        if embeds is not None:
            print("  -> Reusing embeddings from RAM cache (text encoder not loaded).")
        else:
            os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
            cache_path = os.path.join(EMBED_CACHE_DIR, key + ".pt")
            if os.path.exists(cache_path):
                print("  -> Reusing embeddings from disk cache (text encoder not loaded).")
                embeds = torch.load(cache_path, weights_only=False)
            else:
                run_subprocess_logged(
                    "encode_in_subprocess",
                    (MODEL_PATH, config["prompt"], config["negative_prompt"], need_negative, cache_path),
                    cancel_error=CancellationError("Cancelled during text encoding."),
                )
                if not os.path.exists(cache_path):
                    raise RuntimeError("Text encoding failed -- see the subprocess output above.")
                embeds = torch.load(cache_path, weights_only=False)
            if len(_EMBED_MEM_CACHE) >= _EMBED_MEM_CACHE_MAX:
                _EMBED_MEM_CACHE.pop(next(iter(_EMBED_MEM_CACHE)))  # oldest inserted
            _EMBED_MEM_CACHE[key] = embeds

        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = embeds

        if cancel_flag:
            raise CancellationError("Cancelled after text encoding.")

        # --- Stage 2: Models (kept resident in RAM across runs) ---
        onload_device = torch.device("cuda")
        offload_device = torch.device("cpu")

        # `blocks_per_group` and `attention_backend` are baked into the pipeline at
        # build time, so a reused pipe would silently ignore edits to them. Make
        # them part of the cache identity -- changing either rebuilds by itself.
        # A LoRA is baked in the same way (it's merged into the transformer's
        # adapter weights), so it joins the same identity tuple.
        build_opts = (int(config.get("blocks_per_group", 4)),
                      config.get("attention_backend", "native"),
                      config.get("lora_path") or None if LORA_LOADING_ENABLED else None,
                      float(config.get("lora_scale", 1.0)) if LORA_LOADING_ENABLED else None)

        if (_MODEL_CACHE["pipe"] is not None
                and _MODEL_CACHE["path"] == MODEL_PATH
                and _MODEL_CACHE.get("opts") == build_opts):
            pipe = _MODEL_CACHE["pipe"]
            print("--- [2/4] Reusing resident FP8 pipeline (no disk reload) ---")
        else:
            if _MODEL_CACHE["pipe"] is not None and _MODEL_CACHE.get("opts") != build_opts:
                # Say so explicitly -- otherwise an unexpectedly slow run after a
                # settings tweak just looks like the cache broke.
                print(f"  -> Offload settings changed {_MODEL_CACHE['opts']} -> {build_opts}; "
                      "rebuilding pipeline (one slow load).")
                _MODEL_CACHE.update({"pipe": None, "path": None, "opts": None})
                gc.collect()
                torch.cuda.empty_cache()
            print("--- [2/4] Loading FP8 Transformer & Enabling VRAM Protections ---")
            transformer = LTX2VideoTransformer3DModel.from_pretrained(
                MODEL_PATH,
                subfolder="transformer",
                torch_dtype=torch.float8_e4m3fn,
                local_files_only=True,
            )
            patch_transformer_fp8_params(transformer, target_dtype=torch.bfloat16)
            dbg("transformer loaded + fp8 params patched")

            if cancel_flag:
                raise CancellationError("Cancelled during model load.")

            pipe = LTX2Pipeline.from_pretrained(
                MODEL_PATH,
                transformer=transformer,
                text_encoder=None,
                tokenizer=None,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            pipe.set_progress_bar_config(disable=True)

            lora_path = config.get("lora_path") if LORA_LOADING_ENABLED else None
            if config.get("lora_path") and not LORA_LOADING_ENABLED:
                print("  -> LoRA loading is disabled (see LORA_LOADING_ENABLED in "
                      "ltx_engine.py); ignoring lora_path.")
            if lora_path:
                try:
                    pipe.load_lora_weights(lora_path, adapter_name="default_0")
                    # PEFT creates lora_A/lora_B in the same dtype as the layer
                    # they adapt -- fp8 here, since the transformer's weights
                    # rest in fp8 and are only just-in-time upcast inside
                    # dynamic_fp8_linear_forward above. That patch only covers
                    # the base layer's own forward; the LoRA path's own matmul
                    # (lora_B(lora_A(x))) runs unpatched and hits a real fp8
                    # GEMM, which ROCm has no addmm kernel for ("addmm_cuda"
                    # not implemented for 'Float8_e4m3fn'). Cast just the LoRA
                    # adapter weights to bf16 -- same dtype everything else in
                    # the pipeline computes in -- so that path never touches fp8.
                    n_cast = 0
                    for name, param in pipe.transformer.named_parameters():
                        if "lora_" in name and param.dtype in fp8_types:
                            param.data = param.data.to(torch.bfloat16)
                            n_cast += 1
                    if n_cast:
                        dbg(f"cast {n_cast} LoRA adapter tensors fp8 -> bf16")
                    pipe.set_adapters(["default_0"], adapter_weights=[float(config.get("lora_scale", 1.0))])
                    print(f"  -> LoRA loaded: {lora_path} (scale {config.get('lora_scale', 1.0)})")
                except Exception as exc:
                    # A bad or incompatible LoRA file shouldn't take down the
                    # whole run -- continue with the base model instead.
                    print(f"  -> LoRA load failed ({exc}); continuing without it.")

            # "native" = torch SDPA auto-dispatch. On this GPU (gfx1201) the flash,
            # mem-efficient and math kernels are all available, and auto-dispatch
            # picks flash/efficient where the mask allows and falls back safely
            # where it does not. Force "_native_flash" / "_native_efficient" via
            # config only if you want to experiment -- a forced kernel will hard-fail
            # on the masked cross-attention calls.
            if hasattr(pipe.transformer, "set_attention_backend"):
                backend = config.get("attention_backend", "native")
                try:
                    pipe.transformer.set_attention_backend(backend)
                    print(f"  -> Attention backend: {backend}")
                except Exception as exc:
                    print(f"  -> Attention backend '{backend}' unavailable ({exc}); using default.")

            # On a big enough card, skip offload/streaming entirely and just keep
            # the transformer GPU-resident -- see FULL_RESIDENT_VRAM_THRESHOLD_GB.
            _, _, vram_total_gb = hw_monitor.get_gpu_stats()
            full_resident = bool(vram_total_gb) and vram_total_gb >= FULL_RESIDENT_VRAM_THRESHOLD_GB
            if full_resident:
                print(f"  -> {vram_total_gb:.0f}GB VRAM detected (>= "
                      f"{FULL_RESIDENT_VRAM_THRESHOLD_GB:.0f}GB estimated threshold); "
                      "keeping transformer GPU-resident, no offload streaming.")
                pipe.transformer.to(onload_device)
            else:
                # Group offload: 48 transformer blocks, streamed from pinned host RAM.
                # NOTE: `use_stream=True` forces num_blocks_per_group to 1 -- diffusers
                # only supports the prefetch stream at a group size of 1 and logs
                # "Setting it to 1." So `blocks_per_group` below is inert while streams
                # are on. That is the faster arrangement anyway (the transfer overlaps
                # compute instead of stalling it), which is why there is no GUI control
                # for it. It only takes effect if you also set use_stream=False.
                offload_kwargs = dict(
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="block_level",
                    num_blocks_per_group=int(config.get("blocks_per_group", 4)),
                    use_stream=True,
                    record_stream=True,
                    non_blocking=True,
                    low_cpu_mem_usage=False,   # False => pinned (page-locked) host buffers
                )
                if hasattr(pipe.transformer, "enable_group_offload"):
                    pipe.transformer.enable_group_offload(**offload_kwargs)
                else:
                    apply_group_offloading(pipe.transformer, **offload_kwargs)

            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()

            # The decoders are small (VAE 1.4GB + audio VAE 0.1GB + vocoder 0.25GB)
            # and `_execution_device` is cuda either way -- via the transformer's
            # offload hooks in the streamed case, or because it's just .to()'d
            # onto cuda directly in the full-resident case -- so they MUST be on
            # the GPU before pipe() runs -- the old code moved them *after* the
            # call, i.e. never in time for the decode.
            pipe.vae.to(onload_device)
            if getattr(pipe, "audio_vae", None) is not None:
                pipe.audio_vae.to(onload_device)
            if getattr(pipe, "vocoder", None) is not None:
                pipe.vocoder.to(onload_device)

            _MODEL_CACHE.update({"pipe": pipe, "path": MODEL_PATH, "opts": build_opts})

        # VAE tiling. The old `tile_sample_min_size` / `tile_latent_min_size`
        # assignments did nothing at all -- AutoencoderKLLTX2Video has no such
        # attributes, so tiling silently ran at its 512/448 defaults. These are
        # the real knobs, plus framewise (temporal) decoding, which is what
        # actually bounds VRAM on long clips.
        #
        # Deliberately OUTSIDE the cache-hit branch above, unlike blocks_per_group
        # / attention_backend: enable_tiling() just sets attributes on the VAE
        # (cheap, no rebuild needed), so re-applying it on every run means an
        # edit to vae_tile_* in the config takes effect on the very next
        # generation. It used to live inside the "build a fresh pipe" branch
        # only, so editing tile settings while the pipe was resident did
        # nothing -- silently, with no log line, while the README claimed every
        # config key is "read fresh each run".
        if hasattr(pipe.vae, "enable_tiling"):
            tile = int(config.get("vae_tile_size", 512))
            tframes = int(config.get("vae_tile_frames", 24))
            # Strides MUST land on a latent boundary (spatial 32, temporal 8) --
            # bench_vae_tiles.py measured a 3x penalty for a ragged stride, e.g.
            # a 384px tile with an un-snapped 336px stride (10.5 latent px) ran
            # slower than a 512px tile despite decoding fewer tiles. Snap
            # whatever the config asks for rather than trusting it verbatim.
            raw_stride = config.get("vae_tile_stride")
            raw_stride = int(raw_stride) if raw_stride is not None else int(tile * 0.875)
            tile_stride = max(SPATIAL_COMPRESSION,
                              (raw_stride // SPATIAL_COMPRESSION) * SPATIAL_COMPRESSION)
            raw_tstride = config.get("vae_tile_stride_frames", 16)
            tstride = max(TEMPORAL_COMPRESSION,
                          (int(raw_tstride) // TEMPORAL_COMPRESSION) * TEMPORAL_COMPRESSION)
            if tile_stride != raw_stride or tstride != int(raw_tstride):
                print(f"  -> VAE tile stride snapped to a latent boundary: "
                      f"{raw_stride}->{tile_stride}px, {raw_tstride}->{tstride}f")
            pipe.vae.enable_tiling(
                tile_sample_min_height=tile,
                tile_sample_min_width=tile,
                tile_sample_min_num_frames=tframes,
                tile_sample_stride_height=tile_stride,
                tile_sample_stride_width=tile_stride,
                tile_sample_stride_num_frames=tstride,
            )
            dbg(f"VAE tiling: {tile}px tiles, {tile_stride}px stride "
                f"({tile - tile_stride}px overlap), {tframes}f/{tstride}f temporal")
        # Temporal tiling of the decoder; composes with the spatial tiling above.
        pipe.vae.use_framewise_decoding = True
        pipe.vae.use_framewise_encoding = True

        if cancel_flag:
            raise CancellationError("Cancelled before generation.")

        # --- Stage 3: Generation ---
        generation_start_time = time.time()
        # Models stay resident across runs, so the peak counters carry over from
        # previous generations unless they're reset -- "this run" must mean it.
        torch.cuda.reset_peak_memory_stats()
        # duration_head is what makes Auto Duration possible; fall back to the
        # explicit frame count if this checkpoint doesn't ship one.
        use_auto_duration = want_auto_duration and getattr(pipe, "duration_head", None) is not None
        if want_auto_duration and not use_auto_duration:
            print("  -> Auto Duration requested but this checkpoint has no duration_head; using frame count.")
        if use_auto_duration:
            length_call = dict(min_seconds=auto_min_s, max_seconds=auto_max_s)
            print(f"  -> Auto Duration: model picks length within {auto_min_s:.1f}-{auto_max_s:.1f}s.")
        else:
            length_call = dict(num_frames=config["frames"])

        mode_label = ("Image-to-Video" if use_image else "First+Last-Frame-to-Video" if use_flf2v
                      else "Video-to-Video" if use_video else "Text-to-Video")
        len_label = "auto length" if use_auto_duration else f"{config['frames']} frames"
        print(f"--- [3/4] Generating Base Video, {mode_label} ({stage1_w}x{stage1_h}, {len_label}) ---")
        # Under Auto Duration the model picks the length, so config["frames"]
        # was never actually run -- reporting a token count derived from it
        # was reporting a sequence length that doesn't correspond to anything.
        # Use the worst case Auto Duration is allowed to pick instead (same
        # figure the GUI's pre-flight VRAM warning is built on).
        _dbg_frames = int(auto_max_s * float(config["fps"])) if use_auto_duration else int(config["frames"])
        _lat_f = (_dbg_frames - 1) // 8 + 1
        dbg(f"seq={_lat_f * (stage1_h // 32) * (stage1_w // 32):,} tokens "
            f"(latent {_lat_f}x{stage1_h // 32}x{stage1_w // 32}"
            f"{', worst-case under Auto Duration' if use_auto_duration else ''}), "
            f"blocks_per_group={config.get('blocks_per_group', 4)}, seed={config['active_seed']}")
        _passes = guidance_pass_count(use_cfg, use_stg, modality_scale)
        dbg(f"guidance: CFG {'on' if use_cfg else 'off'}"
            f"{f' (scale {cfg_scale}, audio {audio_cfg_scale})' if use_cfg else ''}, "
            f"STG {'on' if use_stg else 'off'}"
            f"{f' (scale {stg_scale}, blocks {stg_blocks})' if use_stg else ''}, "
            f"modality {'on' if use_modality else 'off'}"
            f"{f' (scale {modality_scale})' if use_modality else ''} "
            f"-> {_passes} transformer pass{'es' if _passes > 1 else ''} per step")

        steps_done = [0]
        last_step_at = [generation_start_time]

        def step_callback(pipe_instance, step_index, timestep, callback_kwargs):
            if cancel_flag:
                raise CancellationError("Cancelled by user during diffusion process.")
            steps_done[0] += 1
            done = steps_done[0]
            now = time.time()
            elapsed = now - generation_start_time
            # Delta, not elapsed/done -- a running average smears the (much more
            # expensive) stage-2 steps into stage 1's and hides the real cost.
            step_time = now - last_step_at[0]
            last_step_at[0] = now
            print(f"  --> Completed Step {done}/{total_steps} ({elapsed:.1f}s elapsed)")
            dbg(f"step {done} t={float(timestep):.1f} {step_time:.1f}s this step")
            root.after(0, progress_var.set, done)
            return callback_kwargs

        # Guidance is per-stage now (CFG mode drives stage 1 only), so it is no
        # longer part of shared_call.
        shared_call = dict(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            frame_rate=config["fps"],
            callback_on_step_end=step_callback,
            return_dict=False,
        )

        # One generator threaded through both stages so stage 2 continues the
        # noise stream (this is what the LTX-2.5 reference two-stage recipe does).
        generator = torch.Generator("cuda").manual_seed(config["active_seed"])

        stage1_conditions = None
        with torch.inference_mode():
            if use_image or use_flf2v:
                # Shares the already-loaded/onloaded transformer, VAE, text
                # encoder etc with `pipe` -- this just wraps the same resident
                # component instances in the image-conditioned __call__, no
                # extra weights are loaded or moved.
                #
                # image_crf: LTX-2.5 re-compresses the conditioning image to CRF
                # 18 by default to match training. 0 skips it and keeps the
                # source detail.
                #
                # `None` does NOT mean "model default" here despite what this
                # used to say: diffusers picks the default via
                # resolve_default_image_crf(text_encoder), which keys off the
                # text encoder's config.model_type -- and this pipe's
                # text_encoder is None (we never load it, on purpose, to save
                # 23GB). That resolves to DEFAULT_IMAGE_CRF = 33, the LTX-2.3
                # value, not this checkpoint's 18. So `None` silently
                # over-compressed every i2v conditioning frame. Default to the
                # correct constant explicitly instead of trusting auto-detect.
                crf = config.get("image_crf")
                if crf is None:
                    crf = LTX2_5_IMAGE_CRF
                if use_flf2v:
                    # Start+end frame conditioning needs the keyframe-aware
                    # condition pipeline -- LTX2ImageToVideoPipeline only ever
                    # conditions frame 0. `index=-1` is a *latent* index (see
                    # preprocess_conditions in pipeline_ltx2_condition.py): it
                    # always resolves to the last frame of whatever length
                    # gets generated, so this works under Auto Duration too,
                    # not just a fixed frame count.
                    i2v_pipe = LTX2ConditionPipeline(**pipe.components)
                    stage1_conditions = [
                        LTX2VideoCondition(frames=input_image, index=0, strength=1.0, crf=int(crf)),
                        LTX2VideoCondition(frames=input_end_image, index=-1, strength=1.0, crf=int(crf)),
                    ]
                    stage1 = i2v_pipe(
                        conditions=stage1_conditions,
                        width=stage1_w,
                        height=stage1_h,
                        generator=generator,
                        output_type="latent" if use_upscale else "np",
                        **stage1_schedule,
                        **stage1_guidance,
                        **length_call,
                        **shared_call,
                    )
                else:
                    i2v_pipe = LTX2ImageToVideoPipeline(**pipe.components)
                    stage1 = i2v_pipe(
                        image=input_image,
                        width=stage1_w,
                        height=stage1_h,
                        generator=generator,
                        output_type="latent" if use_upscale else "np",
                        image_crf=int(crf),
                        **stage1_schedule,
                        **stage1_guidance,
                        **length_call,
                        **shared_call,
                    )
            elif use_video:
                # Same "wrap the resident components" pattern as i2v_pipe above --
                # no extra weights loaded, just the condition-aware __call__.
                v2v_pipe = LTX2ConditionPipeline(**pipe.components)
                # `strength` sets a conditioning_mask value, not a denoise
                # amount: 1.0 = mask=1 = "keep fully clean" (source untouched),
                # 0.0 = mask=0 = fully noised (prompt has maximum freedom).
                # That's the OPPOSITE direction from img2img denoise strength --
                # confirmed by reading prepare_latents() in
                # pipeline_ltx2_condition.py, not assumed.
                video_strength = float(config.get("video_strength", 0.05))
                condition = LTX2VideoCondition(frames=input_video_frames, index=0, strength=video_strength)
                stage1 = v2v_pipe(
                    conditions=[condition],
                    width=stage1_w,
                    height=stage1_h,
                    generator=generator,
                    output_type="latent" if use_upscale else "np",
                    **stage1_schedule,
                    **stage1_guidance,
                    **length_call,
                    **shared_call,
                )
            else:
                stage1 = pipe(
                    width=stage1_w,
                    height=stage1_h,
                    generator=generator,
                    output_type="latent" if use_upscale else "np",
                    **stage1_schedule,
                    **stage1_guidance,
                    **length_call,
                    **shared_call,
                )

            if cancel_flag:
                raise CancellationError("Cancelled before video export.")

            if use_upscale:
                # --- Stage 4: 2x latent upsample + short refinement tail ---
                # stage1[0] = denormalised video latents [B, C, F, H, W]
                # stage1[1] = *audio latents* (NOT a waveform) -- the old code fed
                #             these straight to the muxer, which is why "upscale"
                #             produced broken output.
                stage1_latents, audio_latents = stage1[0], stage1[1]

                # Under Auto Duration the realized length is whatever the model
                # picked, not config["frames"], so read it back off the latents
                # (VAE temporal compression is 8: frames = (F - 1) * 8 + 1).
                if stage1_latents.ndim == 5:
                    realized_frames = (stage1_latents.shape[2] - 1) * 8 + 1
                else:
                    realized_frames = config["frames"]

                print(f"--- [4/4] Latent upsample -> {out_w}x{out_h}, then {len(STAGE_2_DISTILLED_SIGMA_VALUES)}-sigma refinement ---")
                # Group-offload uses non_blocking transfers on a separate stream, so
                # stage 1's last block-eviction copies may still be in flight when
                # we start allocating for the upsampler below. Sync first so we're
                # not racing that transfer for VRAM.
                torch.cuda.synchronize()
                latent_upsampler = _build_latent_upsampler(torch).to(onload_device)
                upscale_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)

                upsampled_latents = upscale_pipe(
                    latents=stage1_latents,
                    output_type="latent",
                    return_dict=False,
                )[0]

                dbg(f"upsampled latents {tuple(upsampled_latents.shape)}")

                # Free the upsampler's ~1GB before the (larger) stage-2 denoise.
                latent_upsampler.to(offload_device)
                torch.cuda.empty_cache()
                dbg("upsampler offloaded, entering stage-2 denoise")

                if cancel_flag:
                    raise CancellationError("Cancelled before refinement pass.")

                # Stage 2 infers its resolution from the 5D latents, so no height/width.
                #
                # i2v runs this through i2v_pipe, not the plain T2V `pipe` --
                # using T2V here was silently dropping the image conditioning
                # for the refinement pass. i2v_pipe.prepare_latents() special-
                # cases 5D `latents` input (pipeline_ltx2_image2video.py:798-809):
                # it builds a conditioning_mask marking frame 0 as clean and only
                # re-noises the *other* frames (`noise_scale * (1 - mask)`),
                # matching what the upstream two-stage recipe does
                # (ltx_pipelines/ti2vid_two_stages.py:301-327). The plain T2V
                # pipeline has no such mask and re-noises every frame including
                # 0, so a T2V-refined i2v stage 1 drifts off the source image in
                # a way i2v single-stage never does. No `image=` needed here --
                # that branch of prepare_latents is only reached when `latents`
                # is None, so passing 5D latents skips it entirely (and CRF
                # re-compression along with it, correctly: the image was
                # already conditioned once, in stage 1).
                #
                # LTX2ConditionPipeline (used for v2v, and for i2v when an end
                # frame is set) is NOT like i2v_pipe here -- read directly from
                # pipeline_ltx2_condition.py's prepare_latents(): it always
                # re-derives the conditioning mask from `conditions`, even when
                # 5D `latents` are supplied, so `conditions` must be passed
                # again on this stage-2 call or the keyframe anchoring is
                # silently dropped for the refinement pass.
                #
                # `height`/`width` must go with it. __call__ infers
                # latent_height/width from the 5D `latents` shape for the base
                # grid, but that inferred size is NOT what it uses to
                # preprocess `conditions` -- preprocess_conditions() is called
                # with the raw height/width *parameters* instead (a diffusers
                # quirk/bug, confirmed by reading __call__: it computes the
                # inferred size into local latent_height/width but then still
                # passes the original height/width through to
                # prepare_latents()). Omitting them defaults to 512x768,
                # so the encoded keyframe tokens come out sized for 512x768
                # instead of this run's actual stage-2 resolution, and the
                # sequence-length mismatch throws ValueError at
                # prepare_latents(). Verified against this checkpoint: this
                # exact call raised "Provided latents tensor has shape
                # [1, 1792, 128], but the expected shape is [1, 2688, 128]"
                # (256x256 stage 1 upscaled to 512x512, defaults implying
                # 512x768) before out_w/out_h were added here.
                stage2_pipe = i2v_pipe if (use_image or use_flf2v) else v2v_pipe if use_video else pipe
                stage2_conditions = {}
                if use_flf2v and stage1_conditions is not None:
                    stage2_conditions["conditions"] = stage1_conditions
                    stage2_conditions["height"] = out_h
                    stage2_conditions["width"] = out_w
                elif use_video:
                    stage2_conditions["conditions"] = [condition]
                    stage2_conditions["height"] = out_h
                    stage2_conditions["width"] = out_w
                output = stage2_pipe(
                    num_frames=realized_frames,
                    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                    latents=upsampled_latents,
                    audio_latents=audio_latents,
                    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                    generator=generator,
                    output_type="np",
                    **stage2_conditions,
                    **distilled_guidance,
                    **shared_call,
                )
            else:
                print("--- [4/4] Skipping Upscaler (Native Resolution selected) ---")
                output = stage1

        video = output[0]
        audio = output[1] if len(output) > 1 else None

        # `reserved`, not `allocated`: the caching allocator holds far more than
        # live tensors, and it's the reserved pool that competes for the 16GB.
        dbg(f"decoded video {getattr(video, 'shape', '?')}, peak VRAM this run "
            f"{torch.cuda.max_memory_reserved() / 1024**3:.2f}GB reserved / "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB allocated")
        # Everything after the final denoise step is VAE decode -- on a 2-stage run
        # that has been the majority of wall-clock, so surface it explicitly.
        dbg(f"VAE decode took {time.time() - last_step_at[0]:.1f}s "
            f"(vs {last_step_at[0] - generation_start_time:.1f}s for all {total_steps} steps)")

        print("  --> Exporting final video...")
        mode_tag = "i2v_" if use_image else "flf2v_" if use_flf2v else "v2v_" if use_video else ""
        # Report what was actually produced -- under Auto Duration this is the
        # model's chosen length, not config["frames"].
        final_frames = len(video[0])
        if use_auto_duration:
            print(f"  --> Auto Duration produced {final_frames} frames ({final_frames / float(config['fps']):.2f}s).")
        # Guidance mode goes in the name: without it a same-seed A/B (STG on
        # vs off) writes the identical filename and the second run silently
        # overwrites the first -- destroying the comparison being made.
        guide_tag = ""
        if use_cfg:
            guide_tag += f"_cfg{cfg_scale:g}"
        if use_stg:
            guide_tag += f"_stg{stg_scale:g}"
        # Same reasoning as CFG/STG above: strength is the setting most worth
        # comparing across identical-seed v2v A/Bs (see the strength-curve
        # investigation), so it needs to be in the name too, not just guidance.
        if use_video:
            guide_tag += f"_str{video_strength:g}"
        output_file = (f"output_{mode_tag}{out_w}x{out_h}_{final_frames}f"
                       f"_seed{config['active_seed']}{guide_tag}.mp4")
        # The name above is fully determined by mode/resolution/frames/seed/
        # guidance -- any exact rerun (a second click, an unattended repeat)
        # reproduces it byte-for-byte and would otherwise silently overwrite
        # the earlier file. Append the first free _2, _3, ... suffix instead.
        if os.path.exists(output_file):
            base, ext = os.path.splitext(output_file)
            n = 2
            while os.path.exists(f"{base}_{n}{ext}"):
                n += 1
            output_file = f"{base}_{n}{ext}"
        sample_rate = 24000
        if getattr(pipe, "vocoder", None) is not None:
            sample_rate = getattr(pipe.vocoder.config, "output_sampling_rate", 24000)

        encode_video(
            video[0],  # first video in the batch
            audio=audio[0].float().cpu() if audio is not None else None,
            audio_sample_rate=sample_rate,
            output_path=output_file,
            fps=int(config["fps"]),
        )

        print(f"\nSUCCESS! Video saved as: {output_file}")
        total_elapsed = time.time() - script_start_time
        gen_elapsed = time.time() - generation_start_time
        print(f"Generation pass time: {gen_elapsed:.1f}s ({gen_elapsed/60:.2f}m)")
        print(f"Total time to completion: {total_elapsed:.1f}s ({total_elapsed/60:.2f}m)")

    except CancellationError as e:
        print(f"\n[!] {str(e)}")
        print("[!] Memory is being cleared. Ready for new input.")
    except Exception as e:
        print(f"\n[!] AN ERROR OCCURRED:\n{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # NOTE: `pipe` / the upsampler are deliberately NOT deleted -- they live in
        # _MODEL_CACHE so the next run skips ~18GB of disk reads. Use the
        # "Unload Models" button to drop them.
        gc.collect()
        try:
            # Guarded: if the very first `import torch` above is what raised
            # the exception being handled, this would raise the identical
            # ImportError -- and an exception here would abort the rest of
            # finally, permanently leaving Generate/Cancel disabled with no
            # way to recover short of restarting. Not hypothetical enough to
            # skip guarding for a two-line cost.
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[!] (cleanup warning, continuing anyway: {exc})")

        root.after(0, btn_generate.config, {"state": "normal"})
        root.after(0, btn_cancel.config, {"state": "disabled"})
        print("-" * 60)


def free_resident_models():
    """Drop the cached pipeline/upsampler and hand the RAM+VRAM back.

    The next generation re-reads ~18GB from disk, so this costs a slow first run
    in exchange for the headroom.
    """
    import torch
    ram_before, _ = hw_monitor.get_ram_stats()
    _MODEL_CACHE.update({"pipe": None, "path": None, "opts": None})
    _UPSAMPLER_CACHE.update({"model": None})
    # Embeddings are also RAM-resident; the disk cache still backs them, so
    # dropping these only costs a torch.load on reuse.
    _EMBED_MEM_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # `empty_cache()` only frees *device* memory. Group offload runs with
    # low_cpu_mem_usage=False, i.e. pinned (page-locked) host buffers, and
    # PyTorch's caching host allocator holds those on to reuse them -- so the
    # ~18GB of transformer weights stays resident even after every Python
    # reference is gone. This is the call that actually hands it back.
    host_empty = getattr(torch._C, "_host_emptyCache", None)
    if host_empty is not None:
        host_empty()
    else:
        print("[*] Note: this torch build has no pinned-host cache flush; "
              "some RAM may stay held by the allocator.")
    ram_after, _ = hw_monitor.get_ram_stats()
    print(f"[*] Resident models released (~{max(0.0, ram_before - ram_after):.1f}GB RAM freed). "
          "Next run reloads from disk.")



