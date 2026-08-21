#!/usr/bin/env python3
"""LTX-2.5 generation engine: everything that actually makes a video.

No UI of any kind lives here. Two front-ends drive it:
  generate_video.py  -- Tk control panel
  cli_gen_vid.py     -- headless CLI

`generation_worker()` marshals progress through objects passed in (anything
with .after/.set/.config), which is why the CLI can hand it plain stubs.
"""
import os
import gc
import sys
import time
import json
import glob
import random
import hashlib
import tempfile
import threading
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
MODEL_PATH = os.path.abspath("./local_ltx25_fp8")
# LTX-2.5 model dir: holds the *correct* latent_upsampler config. Don't be
# tempted by Lightricks/ltxv-spatial-upscaler-0.9.7 -- that is an LTX-1 / 0.9.x
# upsampler and is architecturally incompatible with the LTX-2.5 VAE.
BASE_MODEL_PATH = os.path.abspath("./local_ltx25_model")
EMBED_CACHE_DIR = os.path.abspath("./.embed_cache")
# google/gemma-4-E2B-it. The LTX-2.5 checkpoints ship `prompt_enhancer` and
# `processor` as nulls, so the enhancer is a separate download.
ENHANCER_PATH = os.path.abspath("./local_ltx25_enhancer")

# Auto Duration can predict up to 20s; at these resolutions that is far past
# what 16GB of VRAM survives. Hard ceiling on what the model is allowed to pick.
AUTO_DURATION_CAP_S = 6.0

# From the VAE config: spatial_compression_ratio 32, temporal_compression_ratio 8.
# Width/height must be multiples of 32; frame counts must satisfy 8k + 1.
SPATIAL_COMPRESSION = 32
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
VRAM_GB_PER_TOKEN = 1.814e-4
VRAM_HEADROOM = 0.85          # leave 15% for spikes the steady state misses
TOKEN_WARN_FALLBACK = 30000   # if VRAM can't be read: the measured 16GB value


def token_warn_threshold(config=None):
    """Latent-token count above which to warn, scaled to the card actually
    present. Falls back to the measured 16GB figure if VRAM is unreadable."""
    if config and config.get("token_warn_threshold"):
        return int(config["token_warn_threshold"])
    _, _, vram_total = hw_monitor.get_gpu_stats()
    if not vram_total:
        return TOKEN_WARN_FALLBACK
    tokens = (vram_total * VRAM_HEADROOM - VRAM_BASE_GB) / VRAM_GB_PER_TOKEN
    # A card too small to hold the base footprint gets the floor, not a
    # negative threshold -- it will warn on essentially everything, correctly.
    return max(2000, int(tokens))

# --- Process-lifetime caches -------------------------------------------------
# Peak RAM use is ~45GB, so on anything comfortably above that there is no
# reason to re-read 18GB of transformer weights
# (or 23GB of text encoder) from disk on every click of "Generate".
_MODEL_CACHE = {"pipe": None, "transformer": None, "path": None, "opts": None}
_UPSAMPLER_CACHE = {"model": None, "path": None}
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


def latent_tokens(width, height, frames, upscale):
    """Transformer sequence length of the final stage -- the number the VRAM
    warning is built on. Shared by the GUI and the CLI so they can't disagree."""
    scale = 2 if upscale else 1
    lat_f = (frames - 1) // 8 + 1
    return lat_f * ((height * scale) // 32) * ((width * scale) // 32)


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


def _enhance_prompt_inproc(torch, p, image_path):
    """Run the Gemma-4 prompt enhancer. Always called inside a throwaway
    subprocess so the ~10GB enhancer never coexists with the transformer.

    `LTX2Pipeline.enhance_prompt` only reads `.processor`, `.prompt_enhancer`
    and `._execution_device`, so it runs against a bare shell object -- no need
    to load the 23GB text encoder just to rewrite a sentence.
    """
    import types
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

    print("  -> Loading prompt enhancer (Gemma-4 E2B)...")
    pipe_text = types.SimpleNamespace(_execution_device="cpu")
    pipe_text.enhance_prompt = LTX2Pipeline.enhance_prompt.__get__(pipe_text)
    pipe_text.processor = AutoProcessor.from_pretrained(ENHANCER_PATH, local_files_only=True)
    pipe_text.prompt_enhancer = AutoModelForCausalLM.from_pretrained(
        ENHANCER_PATH, dtype=torch.bfloat16, local_files_only=True,
    ).eval()

    system_prompt = LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT if image is not None else LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT

    def run(device):
        with torch.no_grad():
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


def enhance_in_subprocess(p, image_path, out_path):
    """Subprocess target for the standalone '✨ Enhance Now' button."""
    import torch
    with open(out_path, "w") as f:
        f.write(_enhance_prompt_inproc(torch, p, image_path))


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
    torch.save(embeds, out_path)
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

        import warnings
        import torch
        import torch.nn.functional as F
        import diffusers
        from diffusers import (
            LTX2Pipeline,
            LTX2VideoTransformer3DModel,
            LTX2LatentUpsamplePipeline,
            LTX2ImageToVideoPipeline,
        )
        from diffusers.hooks import apply_group_offloading
        from diffusers.pipelines.ltx2.utils import (
            DISTILLED_SIGMA_VALUES,
            STAGE_2_DISTILLED_SIGMA_VALUES,
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

        input_image = None
        if use_image:
            from PIL import Image

            image_path = config.get("image_path", "")
            if not image_path or not os.path.exists(image_path):
                raise ValueError(f"Image-to-video mode selected but no valid image path was given: '{image_path}'")
            input_image = Image.open(image_path).convert("RGB")

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

        if use_cfg:
            stage1_guidance = dict(
                guidance_scale=cfg_scale,
                audio_guidance_scale=cfg_scale,
                stg_scale=0.0,          # STG is a third forward pass; not worth it here
                audio_stg_scale=0.0,
                modality_scale=1.0,
                audio_modality_scale=1.0,
            )
            stage1_schedule = dict(num_inference_steps=cfg_steps)
            need_negative = True        # the negative prompt is finally encoded and used
        else:
            stage1_guidance = distilled_guidance
            stage1_schedule = dict(sigmas=DISTILLED_SIGMA_VALUES)
            need_negative = False       # CFG off => negative branch is never evaluated

        stage1_w, stage1_h = int(config["width"]), int(config["height"])
        out_w, out_h = (stage1_w * 2, stage1_h * 2) if use_upscale else (stage1_w, stage1_h)

        # Auto Duration: with a `duration_head` present the model predicts clip
        # length from the prompt when `num_frames` is omitted. Left unbounded it
        # will happily pick up to 20s, which no 16GB card survives at these
        # resolutions -- so max_seconds is clamped hard (see AUTO_DURATION_CAP_S).
        want_auto_duration = bool(config.get("auto_duration"))
        auto_min_s = float(config.get("auto_min_seconds", 2.0))
        auto_max_s = min(float(config.get("auto_max_seconds", 5.0)), AUTO_DURATION_CAP_S)
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
        build_opts = (int(config.get("blocks_per_group", 4)),
                      config.get("attention_backend", "native"))

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
                _MODEL_CACHE.update({"pipe": None, "transformer": None, "path": None, "opts": None})
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

            # VAE tiling. The old `tile_sample_min_size` / `tile_latent_min_size`
            # assignments did nothing at all -- AutoencoderKLLTX2Video has no such
            # attributes, so tiling silently ran at its 512/448 defaults. These are
            # the real knobs, plus framewise (temporal) decoding, which is what
            # actually bounds VRAM on long clips.
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            # Tile geometry is config-driven: VAE decode dominates wall-clock on a
            # 2-stage run (~2/3 of it), and bigger tiles mean fewer overlap-blend
            # passes. Raising these trades VRAM and ring-timeout headroom for speed.
            if hasattr(pipe.vae, "enable_tiling"):
                tile = int(config.get("vae_tile_size", 512))
                tile_stride = int(config.get("vae_tile_stride", int(tile * 0.875)))
                pipe.vae.enable_tiling(
                    tile_sample_min_height=tile,
                    tile_sample_min_width=tile,
                    tile_sample_min_num_frames=int(config.get("vae_tile_frames", 24)),
                    tile_sample_stride_height=tile_stride,
                    tile_sample_stride_width=tile_stride,
                    tile_sample_stride_num_frames=int(config.get("vae_tile_stride_frames", 16)),
                )
                dbg(f"VAE tiling: {tile}px tiles, {tile_stride}px stride "
                    f"({tile - tile_stride}px overlap)")
            # Temporal tiling of the decoder; composes with the spatial tiling above.
            pipe.vae.use_framewise_decoding = True
            pipe.vae.use_framewise_encoding = True

            # The decoders are small (VAE 1.4GB + audio VAE 0.1GB + vocoder 0.25GB)
            # and `_execution_device` is cuda because of the transformer's offload
            # hooks, so they MUST be on the GPU before pipe() runs -- the old code
            # moved them *after* the call, i.e. never in time for the decode.
            pipe.vae.to(onload_device)
            if getattr(pipe, "audio_vae", None) is not None:
                pipe.audio_vae.to(onload_device)
            if getattr(pipe, "vocoder", None) is not None:
                pipe.vocoder.to(onload_device)

            _MODEL_CACHE.update({"pipe": pipe, "transformer": transformer, "path": MODEL_PATH,
                                 "opts": build_opts})

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

        mode_label = "Image-to-Video" if use_image else "Text-to-Video"
        len_label = "auto length" if use_auto_duration else f"{config['frames']} frames"
        print(f"--- [3/4] Generating Base Video, {mode_label} ({stage1_w}x{stage1_h}, {len_label}) ---")
        _lat_f = (int(config["frames"]) - 1) // 8 + 1
        dbg(f"seq={_lat_f * (stage1_h // 32) * (stage1_w // 32):,} tokens "
            f"(latent {_lat_f}x{stage1_h // 32}x{stage1_w // 32}), "
            f"blocks_per_group={config.get('blocks_per_group', 4)}, seed={config['active_seed']}")

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

        with torch.inference_mode():
            if use_image:
                # Shares the already-loaded/onloaded transformer, VAE, text
                # encoder etc with `pipe` -- this just wraps the same resident
                # component instances in the image-conditioned __call__, no
                # extra weights are loaded or moved.
                i2v_pipe = LTX2ImageToVideoPipeline(**pipe.components)
                # image_crf: LTX-2.5 re-compresses the conditioning image to CRF
                # 18 by default to match training. 0 skips it and keeps the
                # source detail; None uses the model default.
                crf = config.get("image_crf", None)
                stage1 = i2v_pipe(
                    image=input_image,
                    width=stage1_w,
                    height=stage1_h,
                    generator=generator,
                    output_type="latent" if use_upscale else "np",
                    **({} if crf is None else {"image_crf": int(crf)}),
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
                output = pipe(
                    num_frames=realized_frames,
                    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                    latents=upsampled_latents,
                    audio_latents=audio_latents,
                    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                    generator=generator,
                    output_type="np",
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
        mode_tag = "i2v_" if use_image else ""
        # Report what was actually produced -- under Auto Duration this is the
        # model's chosen length, not config["frames"].
        final_frames = len(video[0])
        if use_auto_duration:
            print(f"  --> Auto Duration produced {final_frames} frames ({final_frames / float(config['fps']):.2f}s).")
        output_file = f"output_{mode_tag}{out_w}x{out_h}_{final_frames}f_seed{config['active_seed']}.mp4"
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
        try:
            del upscale_pipe
        except Exception:
            pass

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    _MODEL_CACHE.update({"pipe": None, "transformer": None, "path": None, "opts": None})
    _UPSAMPLER_CACHE.update({"model": None, "path": None})
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



