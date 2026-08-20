#!/usr/bin/env python3
import os
import gc
import sys
import time
import json
import glob
import random
import hashlib
import threading
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

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
os.environ["MIOPEN_FIND_MODE"] = "1"
os.environ["AMD_DIRECT_DISPATCH"] = "1"
os.environ["HIP_FORCE_DEV_KERN_LAZY_COMPILE"] = "0"
os.environ["TORCH_ROCM_AOTXN_ENABLE"] = "1"
os.environ["AMD_SERIALIZE_KERNEL"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
# 124GB RAM / 16 threads: let the CPU-side maths (text encode, pinning,
# numpy postprocess, ffmpeg staging) actually use the box.
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

CONFIG_FILE = "ltx2_config.json"
MODEL_PATH = os.path.abspath("./local_ltx25_fp8")
# LTX-2.5 model dir: holds the *correct* latent_upsampler config (the standalone
# ./local_ltx25_upscaler dir is an LTX-1 / 0.9.x upsampler and is NOT compatible).
BASE_MODEL_PATH = os.path.abspath("./local_ltx25_model")
UPSCALER_PATH = os.path.abspath("./local_ltx25_upscaler")  # legacy, unused
EMBED_CACHE_DIR = os.path.abspath("./.embed_cache")

# Auto Duration can predict up to 20s; at these resolutions that is far past
# what 16GB of VRAM survives. Hard ceiling on what the model is allowed to pick.
AUTO_DURATION_CAP_S = 6.0

# --- Process-lifetime caches -------------------------------------------------
# With 124GB of RAM there is no reason to re-read 18GB of transformer weights
# (or 23GB of text encoder) from disk on every click of "Generate".
_MODEL_CACHE = {"pipe": None, "transformer": None, "path": None}
_UPSAMPLER_CACHE = {"model": None, "path": None}
_EMBED_MEM_CACHE = {}

cancel_flag = False

class CancellationError(Exception):
    pass

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

class TextRedirector:
    def __init__(self, widget):
        self.widget = widget
    def write(self, text):
        self.widget.after(0, self._write, text)
    def _write(self, text):
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")
    def flush(self):
        pass

# =========================================================================
# 3. Stage 1 Subprocess: Text Encoding 
# =========================================================================
def _embed_cache_key(model_path, p, np_text, need_negative):
    h = hashlib.sha256()
    for part in (model_path, p, np_text or "", "neg" if need_negative else "noneg"):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


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

    NOTE: ./local_ltx25_upscaler is an *LTX-1 / 0.9.x* upsampler
    (`LTXLatentUpsamplerModel` + `AutoencoderKLLTXVideo`) and is architecturally
    incompatible with the LTX-2.5 VAE. The real LTX-2.5 weights live in
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
    global cancel_flag

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
        diffusers.logging.set_verbosity_error()

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
        need_negative = False  # CFG off => negative branch is never evaluated

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

        total_steps = len(DISTILLED_SIGMA_VALUES) + (len(STAGE_2_DISTILLED_SIGMA_VALUES) if use_upscale else 0)
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
                p_proc = mp.Process(
                    target=encode_in_subprocess,
                    args=(MODEL_PATH, config["prompt"], config["negative_prompt"], need_negative, cache_path),
                )
                p_proc.start()
                while p_proc.is_alive():
                    p_proc.join(timeout=0.5)
                    if cancel_flag:
                        p_proc.terminate()
                        p_proc.join()
                        raise CancellationError("Cancelled during text encoding.")
                if not os.path.exists(cache_path):
                    raise RuntimeError("Text encoding failed. Check subprocess output.")
                embeds = torch.load(cache_path, weights_only=False)
            _EMBED_MEM_CACHE[key] = embeds

        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = embeds

        if cancel_flag:
            raise CancellationError("Cancelled after text encoding.")

        # --- Stage 2: Models (kept resident in RAM across runs) ---
        onload_device = torch.device("cuda")
        offload_device = torch.device("cpu")

        if _MODEL_CACHE["pipe"] is not None and _MODEL_CACHE["path"] == MODEL_PATH:
            pipe = _MODEL_CACHE["pipe"]
            print("--- [2/4] Reusing resident FP8 pipeline (no disk reload) ---")
        else:
            print("--- [2/4] Loading FP8 Transformer & Enabling VRAM Protections ---")
            transformer = LTX2VideoTransformer3DModel.from_pretrained(
                MODEL_PATH,
                subfolder="transformer",
                torch_dtype=torch.float8_e4m3fn,
                local_files_only=True,
            )
            patch_transformer_fp8_params(transformer, target_dtype=torch.bfloat16)

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

            # Group offload: 48 transformer blocks. 2 blocks/group was a *RAM*-era
            # setting; with 124GB we can pin the whole 18GB checkpoint and use
            # bigger groups + stream prefetch + record_stream (no per-group sync).
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
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling(
                    tile_sample_min_height=512,
                    tile_sample_min_width=512,
                    tile_sample_min_num_frames=24,
                    tile_sample_stride_height=448,
                    tile_sample_stride_width=448,
                    tile_sample_stride_num_frames=16,
                )
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

            _MODEL_CACHE.update({"pipe": pipe, "transformer": transformer, "path": MODEL_PATH})

        if cancel_flag:
            raise CancellationError("Cancelled before generation.")

        # --- Stage 3: Generation ---
        generation_start_time = time.time()
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

        steps_done = [0]

        def step_callback(pipe_instance, step_index, timestep, callback_kwargs):
            if cancel_flag:
                raise CancellationError("Cancelled by user during diffusion process.")
            steps_done[0] += 1
            done = steps_done[0]
            elapsed = time.time() - generation_start_time
            print(f"  --> Completed Step {done}/{total_steps} ({elapsed:.1f}s elapsed)")
            root.after(0, progress_var.set, done)
            return callback_kwargs

        shared_call = dict(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            frame_rate=config["fps"],
            callback_on_step_end=step_callback,
            return_dict=False,
            **distilled_guidance,
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
                    sigmas=DISTILLED_SIGMA_VALUES,
                    generator=generator,
                    output_type="latent" if use_upscale else "np",
                    **({} if crf is None else {"image_crf": int(crf)}),
                    **length_call,
                    **shared_call,
                )
            else:
                stage1 = pipe(
                    width=stage1_w,
                    height=stage1_h,
                    sigmas=DISTILLED_SIGMA_VALUES,
                    generator=generator,
                    output_type="latent" if use_upscale else "np",
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

                # Free the upsampler's ~1GB before the (larger) stage-2 denoise.
                latent_upsampler.to(offload_device)
                torch.cuda.empty_cache()

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
                    **shared_call,
                )
            else:
                print("--- [4/4] Skipping Upscaler (Native Resolution selected) ---")
                output = stage1

        video = output[0]
        audio = output[1] if len(output) > 1 else None

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
        # "Free Models" button to drop them.
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
    """Drop the cached pipeline/upsampler and hand the RAM+VRAM back."""
    import torch
    _MODEL_CACHE.update({"pipe": None, "transformer": None, "path": None})
    _UPSAMPLER_CACHE.update({"model": None, "path": None})
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[*] Resident models released.")

# =========================================================================
# 5. Main Application / GUI
# =========================================================================
def main():
    def load_saved_config():
        defaults = {
            "prompt": "A high-quality cinematic shot of a classic sports car driving along a coastal highway at sunset, vibrant orange horizon, clear ocean view.",
            "negative_prompt": "",
            # Reference LTX-2.5 two-stage base resolution; stage 2 emits 1920x1088.
            "width": 960,
            "height": 544,
            "frames": 121,
            "fps": 24.0,
            "seed": "42",
            "upscale": False,
            "mode": "text2video",
            "image_path": "",
            # Auto Duration: let the model pick clip length from the prompt.
            # Capped at AUTO_DURATION_CAP_S regardless of what's set here.
            "auto_duration": False,
            "auto_min_seconds": 2.0,
            "auto_max_seconds": 5.0,
            # Conditioning-image compression for image-to-video. null = model
            # default (CRF 18 on LTX-2.5), 0 = keep full source detail.
            "image_crf": None,
            # 48 transformer blocks. 4 => 12 offload groups; raise to 6-8 for a bit
            # more speed at the cost of VRAM, drop to 2 if stage 2 OOMs.
            "blocks_per_group": 4
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    defaults.update(json.load(f))
            except: pass
        return defaults

    def save_config(config):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except: pass

    config = load_saved_config()
    
    root = tk.Tk()
    root.title("🎬 LTX-2.5 Control Panel")
    root.geometry("740x980")
    root.resizable(False, False)
    
    style = ttk.Style(root)
    style.theme_use('clam')
    
    # Telemetry Panel
    telemetry_frame = tk.Frame(root, bg="#111111", pady=8, padx=12)
    telemetry_frame.pack(fill=tk.X)
    
    primary_metrics = tk.Frame(telemetry_frame, bg="#111111")
    primary_metrics.pack(fill=tk.X)
    
    lbl_cpu = tk.Label(primary_metrics, text="CPU: --%", bg="#111111", fg="#ffffff", font=("Consolas", 10, "bold"), width=15, anchor=tk.W)
    lbl_cpu.pack(side=tk.LEFT)
    
    lbl_ram = tk.Label(primary_metrics, text="RAM: -- / -- GB", bg="#111111", fg="#ffffff", font=("Consolas", 10, "bold"), width=20)
    lbl_ram.pack(side=tk.LEFT, expand=True)
    
    lbl_gpu = tk.Label(primary_metrics, text="GPU: --%", bg="#111111", fg="#00ffcc", font=("Consolas", 10, "bold"), width=12)
    lbl_gpu.pack(side=tk.LEFT, expand=True)
    
    lbl_vram = tk.Label(primary_metrics, text="VRAM: -- / -- GB", bg="#111111", fg="#00ffcc", font=("Consolas", 10, "bold"), width=22, anchor=tk.E)
    lbl_vram.pack(side=tk.RIGHT)

    cores_box = tk.Frame(telemetry_frame, bg="#222222", padx=6, pady=5, relief=tk.SUNKEN, bd=1)
    cores_box.pack(fill=tk.X, pady=(6, 0))
    
    lbl_cores_text = tk.Label(
        cores_box, 
        text="Reading CPU Core states...", 
        bg="#222222", 
        fg="#00ff88", 
        font=("Consolas", 8, "bold"), 
        justify=tk.LEFT,
        anchor=tk.W
    )
    lbl_cores_text.pack(fill=tk.X)

    def update_telemetry():
        cpu_avg, cores = hw_monitor.get_cpu_stats()
        ram_used, ram_total = hw_monitor.get_ram_stats()
        gpu_usage, vram_used, vram_total = hw_monitor.get_gpu_stats()
        
        lbl_cpu.config(text=f"CPU: {cpu_avg:.0f}% (Avg)")
        lbl_ram.config(text=f"RAM: {ram_used:.1f}/{ram_total:.1f} GB")
        
        if gpu_usage is None:
            lbl_gpu.config(text="GPU: N/A")
            lbl_vram.config(text="VRAM: N/A")
        else:
            lbl_gpu.config(text=f"GPU: {gpu_usage}%")
            lbl_vram.config(text=f"VRAM: {vram_used:.1f}/{vram_total:.1f} GB")
            
        if cores:
            row_chunks = []
            for i in range(0, len(cores), 8):
                chunk = cores[i:i + 8]
                row_str = "   ".join([f"C{i + idx:02d}:{int(p):2d}%" for idx, p in enumerate(chunk)])
                row_chunks.append(row_str)
            lbl_cores_text.config(text="\n".join(row_chunks))
            
        root.after(500, update_telemetry)

    update_telemetry()
    
    # Main Form
    main_frame = ttk.Frame(root, padding="15")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # --- Generation mode ---
    mode_frame = ttk.LabelFrame(main_frame, text=" Generation Mode ", padding="8")
    mode_frame.pack(fill=tk.X, pady=(0, 12))

    mode_var = tk.StringVar(value=config.get("mode", "text2video"))
    image_path_var = tk.StringVar(value=config.get("image_path", ""))

    mode_row = ttk.Frame(mode_frame)
    mode_row.pack(fill=tk.X)
    ttk.Radiobutton(mode_row, text="Text → Video", variable=mode_var, value="text2video").pack(side=tk.LEFT)
    ttk.Radiobutton(mode_row, text="Image → Video", variable=mode_var, value="image2video").pack(side=tk.LEFT, padx=(12, 0))

    image_row = ttk.Frame(mode_frame)
    image_row.pack(fill=tk.X, pady=(6, 0))
    lbl_image = ttk.Label(image_row, textvariable=image_path_var, foreground="#666666")

    def browse_image():
        path = filedialog.askopenfilename(
            title="Select conditioning image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if path:
            image_path_var.set(path)

    btn_browse = ttk.Button(image_row, text="📁 Choose Image...", command=browse_image)

    def on_mode_switch(*_args):
        if mode_var.get() == "image2video":
            btn_browse.pack(side=tk.LEFT)
            lbl_image.pack(side=tk.LEFT, padx=(8, 0))
        else:
            btn_browse.pack_forget()
            lbl_image.pack_forget()

    mode_var.trace_add("write", on_mode_switch)
    on_mode_switch()

    ttk.Label(main_frame, text="Positive Prompt:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
    text_prompt = tk.Text(main_frame, height=3, wrap=tk.WORD, font=("Arial", 10))
    text_prompt.pack(fill=tk.X, pady=(0, 12))
    text_prompt.insert(tk.END, config['prompt'])
    
    np_header = ttk.Frame(main_frame)
    np_header.pack(fill=tk.X)
    ttk.Label(
        np_header,
        text="Negative Prompt (unused: distilled schedule runs guidance-free):",
        font=("Arial", 10, "bold"),
    ).pack(side=tk.LEFT)
    
    text_np = tk.Text(main_frame, height=3, wrap=tk.WORD, font=("Arial", 10))
    
    def reset_np():
        text_np.delete("1.0", tk.END)
        text_np.insert(tk.END, "Loading default prompt...")
        root.update() 
        try:
            from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
            real_default = DEFAULT_NEGATIVE_PROMPT
        except ImportError:
            real_default = "worst quality, inconsistent, deformed, blurry, watermark"
        text_np.delete("1.0", tk.END)
        text_np.insert(tk.END, real_default)
        
    ttk.Button(np_header, text="↺ Reset Default", command=reset_np).pack(side=tk.RIGHT)
    text_np.pack(fill=tk.X, pady=(4, 12))
    text_np.insert(tk.END, config.get('negative_prompt', ""))
        
    settings_frame = ttk.Frame(main_frame)
    settings_frame.pack(fill=tk.X, pady=(0, 12))
    
    res_frame = ttk.LabelFrame(settings_frame, text=" Resolution & Quality ", padding="8")
    res_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
    
    res_var = tk.StringVar(value=f"{config['width']}x{config['height']}")
    res_combo = ttk.Combobox(res_frame, textvariable=res_var, state="readonly", width=18)
    res_combo['values'] = (
        "1280x704 (High)",
        "1024x576 (Medium)",
        "960x544 (2-stage base)",
        "768x512 (Low)",
        "Custom",
    )
    res_combo.pack(pady=(0, 4))
    
    custom_frame = ttk.Frame(res_frame)
    custom_frame.pack()
    ttk.Label(custom_frame, text="W:").pack(side=tk.LEFT)
    entry_w = ttk.Entry(custom_frame, width=5)
    entry_w.pack(side=tk.LEFT, padx=(2, 8))
    entry_w.insert(0, str(config['width']))
    ttk.Label(custom_frame, text="H:").pack(side=tk.LEFT)
    entry_h = ttk.Entry(custom_frame, width=5)
    entry_h.pack(side=tk.LEFT, padx=(2, 0))
    entry_h.insert(0, str(config['height']))

    # The Restored Upscaler Checkbox
    upscale_var = tk.BooleanVar(value=config.get('upscale', False))
    ttk.Checkbutton(
        res_frame,
        text="2-stage: 2x latent upscale + refine\n(output = 2x the size above)",
        variable=upscale_var,
    ).pack(pady=(5, 0))
    
    def on_res_select(event):
        val = res_combo.get()
        entry_w.config(state="normal"); entry_h.config(state="normal")
        if val != "Custom":
            w, h = val.split(" ")[0].split("x")
            entry_w.delete(0, tk.END); entry_w.insert(0, w)
            entry_h.delete(0, tk.END); entry_h.insert(0, h)
            entry_w.config(state="disabled"); entry_h.config(state="disabled")
            
    res_combo.bind("<<ComboboxSelected>>", on_res_select)
    if f"{config['width']}x{config['height']}" not in [v.split(" ")[0] for v in res_combo['values'][:-1]]:
        res_combo.set("Custom")
    on_res_select(None)
    
    time_frame = ttk.LabelFrame(settings_frame, text=" Timing & Length ", padding="8")
    time_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    fps_sub = ttk.Frame(time_frame)
    fps_sub.pack(anchor=tk.W)
    ttk.Label(fps_sub, text="FPS:").pack(side=tk.LEFT)
    entry_fps = ttk.Entry(fps_sub, width=6)
    entry_fps.pack(side=tk.LEFT, padx=4)
    entry_fps.insert(0, str(config['fps']))
    
    length_type = tk.StringVar(value="frames")
    entry_len = ttk.Entry(time_frame, width=8)
    base_frames = float(config['frames'])
    
    def get_safe_fps():
        try: return max(1.0, float(entry_fps.get()))
        except: return 24.0

    def on_mode_change():
        nonlocal base_frames
        fps = get_safe_fps()
        if length_type.get() == "seconds":
            try: base_frames = float(entry_len.get())
            except: pass
            entry_len.delete(0, tk.END); entry_len.insert(0, f"{base_frames/fps:.2f}".rstrip('0').rstrip('.'))
        else:
            try: base_frames = round(float(entry_len.get()) * fps)
            except: pass
            entry_len.delete(0, tk.END); entry_len.insert(0, str(int(base_frames)))

    def on_fps_typing(event=None):
        if length_type.get() == "seconds":
            fps = get_safe_fps()
            entry_len.delete(0, tk.END); entry_len.insert(0, f"{base_frames/fps:.2f}".rstrip('0').rstrip('.'))
            
    def on_len_typing(event=None):
        nonlocal base_frames
        fps = get_safe_fps()
        try:
            val = float(entry_len.get())
            base_frames = val if length_type.get() == "frames" else val * fps
        except: pass

    mode_sub = ttk.Frame(time_frame)
    mode_sub.pack(anchor=tk.W, pady=4)
    ttk.Radiobutton(mode_sub, text="Frames", variable=length_type, value="frames", command=on_mode_change).pack(side=tk.LEFT)
    ttk.Radiobutton(mode_sub, text="Seconds", variable=length_type, value="seconds", command=on_mode_change).pack(side=tk.LEFT, padx=(5,0))
    entry_len.pack(anchor=tk.W, padx=2)
    entry_len.insert(0, str(config['frames']))
    entry_fps.bind("<KeyRelease>", on_fps_typing)
    entry_len.bind("<KeyRelease>", on_len_typing)

    # Auto Duration: model picks the length, capped for VRAM safety.
    auto_dur_var = tk.BooleanVar(value=config.get("auto_duration", False))

    def on_auto_toggle():
        on_auto = auto_dur_var.get()
        entry_len.config(state="disabled" if on_auto else "normal")
        if on_auto:
            lbl_auto_max.pack(side=tk.LEFT)
            entry_auto_max.pack(side=tk.LEFT, padx=(2, 0))
        else:
            lbl_auto_max.pack_forget()
            entry_auto_max.pack_forget()

    ttk.Checkbutton(
        time_frame,
        text=f"Auto Duration (max {AUTO_DURATION_CAP_S:.0f}s)",
        variable=auto_dur_var,
        command=on_auto_toggle,
    ).pack(anchor=tk.W, pady=(6, 0))

    auto_row = ttk.Frame(time_frame)
    auto_row.pack(anchor=tk.W, pady=(2, 0))
    lbl_auto_max = ttk.Label(auto_row, text="max s:")
    entry_auto_max = ttk.Entry(auto_row, width=4)
    entry_auto_max.insert(0, str(config.get("auto_max_seconds", 5.0)))
    on_auto_toggle()
    
    seed_frame = ttk.Frame(main_frame)
    seed_frame.pack(fill=tk.X, pady=(0, 12))
    ttk.Label(seed_frame, text="Seed ('r' for Random):", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
    entry_seed = ttk.Entry(seed_frame, width=15)
    entry_seed.pack(side=tk.LEFT, padx=10)
    entry_seed.insert(0, str(config['seed']))

    progress_var = tk.IntVar(value=0)
    progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=8, mode='determinate')
    progress_bar.pack(fill=tk.X, pady=(0, 8))
    
    log_text = scrolledtext.ScrolledText(main_frame, height=10, state="disabled", bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
    log_text.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
    
    sys.stdout = TextRedirector(log_text)
    sys.stderr = TextRedirector(log_text)

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill=tk.X)
    
    def start_generation():
        global cancel_flag
        cancel_flag = False 
        
        try:
            p = text_prompt.get("1.0", tk.END).strip()
            if not p:
                messagebox.showerror("Error", "Positive prompt cannot be empty.")
                return
                
            if mode_var.get() == "image2video":
                img = image_path_var.get().strip()
                if not img or not os.path.exists(img):
                    messagebox.showerror("Error", "Image-to-Video mode needs a valid image. Click 'Choose Image...'.")
                    return

            np_val = text_np.get("1.0", tk.END).strip()
            if not np_val:
                try:
                    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
                    np_val = DEFAULT_NEGATIVE_PROMPT
                except ImportError:
                    np_val = "worst quality, inconsistent, deformed, blurry, watermark"
                    
            w_adj = max(256, round(int(entry_w.get()) / 32) * 32)
            h_adj = max(256, round(int(entry_h.get()) / 32) * 32)
            fps = float(entry_fps.get())
            val = float(entry_len.get())
            target_frames = int(val * fps) if length_type.get() == "seconds" else int(val)
            aligned_frames = (max(1, round((target_frames - 1) / 8)) * 8) + 1
            
            s_val = entry_seed.get().strip().lower()
            active_seed = random.randint(0, 2**32 - 1) if s_val == 'r' else int(s_val)

            try:
                auto_max_val = min(float(entry_auto_max.get()), AUTO_DURATION_CAP_S)
            except ValueError:
                auto_max_val = AUTO_DURATION_CAP_S

            # VRAM sanity check. The transformer sequence length is
            # latent_frames * (H/32) * (W/32); attention cost is quadratic in it.
            # ~50k tokens is about where 16GB of VRAM stops being comfortable.
            # Under Auto Duration the model picks the length, so size the check
            # against the worst case it's allowed to choose.
            scale = 2 if upscale_var.get() else 1
            check_frames = int(auto_max_val * fps) if auto_dur_var.get() else aligned_frames
            latent_frames = (check_frames - 1) // 8 + 1
            tokens = latent_frames * ((h_adj * scale) // 32) * ((w_adj * scale) // 32)
            if tokens > 50000:
                if not messagebox.askokcancel(
                    "Large sequence",
                    f"Final stage would run {tokens:,} latent tokens "
                    f"({w_adj*scale}x{h_adj*scale}, {check_frames} frames"
                    f"{' worst-case under Auto Duration' if auto_dur_var.get() else ''}).\n\n"
                    "Above ~50,000 tokens this is likely to exhaust 16GB of VRAM or trip "
                    "the AMDGPU ring-timeout watchdog.\n\nContinue anyway?",
                ):
                    return


            config.update({
                'prompt': p, 'negative_prompt': np_val,
                'width': w_adj, 'height': h_adj, 'fps': fps, 'frames': aligned_frames, 'seed': s_val,
                'active_seed': active_seed,
                'upscale': upscale_var.get(),
                'mode': mode_var.get(),
                'image_path': image_path_var.get().strip(),
                'auto_duration': auto_dur_var.get(),
                'auto_max_seconds': auto_max_val,
            })
            save_config(config)
            
            btn_generate.config(state="disabled")
            btn_cancel.config(state="normal")
            progress_var.set(0)
            
            thread = threading.Thread(
                target=generation_worker,
                args=(config, root, progress_var, progress_bar, btn_generate, btn_cancel),
            )
            thread.daemon = True
            thread.start()
            
        except ValueError:
            messagebox.showerror("Error", "Please ensure numbers are valid.")
            
    def cancel_generation():
        global cancel_flag
        cancel_flag = True
        btn_cancel.config(state="disabled")
        print("\n[!] Cancelling... waiting for current step to yield.")

    btn_generate = ttk.Button(btn_frame, text="🚀 Generate Video", command=start_generation)
    btn_generate.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 4))
    
    btn_cancel = ttk.Button(btn_frame, text="🛑 Cancel", command=cancel_generation, state="disabled")
    btn_cancel.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(4, 4))

    # Models stay resident in RAM between runs (18GB transformer + 1.8GB decoders).
    # On a 124GB box that is free speed; this button gives it back if needed.
    btn_free = ttk.Button(btn_frame, text="🧹 Free Models", command=free_resident_models)
    btn_free.pack(side=tk.RIGHT, ipady=8, padx=(4, 0))

    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (740 // 2)
    y = (root.winfo_screenheight() // 2) - (980 // 2)
    root.geometry(f"+{x}+{y}")
    
    print("Welcome to LTX-2.5 Control Panel.")
    print("System ready. Modify settings above and click Generate Video.")
    
    root.mainloop()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
