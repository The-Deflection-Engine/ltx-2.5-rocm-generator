#!/usr/bin/env python3
import os
import gc
import sys
import time
import json
import random
import threading
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# =========================================================================
# 1. Environment Configuration
# =========================================================================
os.environ["PYTORCH_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
os.environ["MIOPEN_FIND_MODE"] = "1"
os.environ["AMD_DIRECT_DISPATCH"] = "1"
os.environ["HIP_FORCE_DEV_KERN_LAZY_COMPILE"] = "0"
os.environ["TORCH_ROCM_AOTXN_ENABLE"] = "1"
os.environ["AMD_SERIALIZE_KERNEL"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

CONFIG_FILE = "ltx2_config.json"
MODEL_PATH = os.path.abspath("./local_ltx25_fp8")
UPSCALER_PATH = os.path.abspath("./local_ltx25_upscaler")

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
        
        for i in range(10):
            path = f"/sys/class/drm/card{i}/device"
            vram_path = os.path.join(path, "mem_info_vram_total")
            if os.path.exists(vram_path):
                try:
                    with open(vram_path, "r") as f:
                        if int(f.read().strip()) > 0:
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
def encode_in_subprocess(model_path, p, np):
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
        embeds = pipe_text.encode_prompt(prompt=p, negative_prompt=np, device="cpu")
    torch.save(embeds, "tmp_embeds.pt")
    print("  -> Encoding complete. Terminating process to release RAM.")

# =========================================================================
# 4. Background Generation Thread
# =========================================================================
def generation_worker(config, root, progress_var, btn_generate, btn_cancel):
    global cancel_flag
    
    try:
        print("\n[*] Initializing PyTorch and ROCm backends...")
        script_start_time = time.time()
        
        import warnings
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import diffusers
        from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel, LTXLatentUpsamplePipeline
        from diffusers.hooks import apply_group_offloading
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        from diffusers.utils import encode_video

        warnings.filterwarnings("ignore", category=FutureWarning)
        diffusers.logging.set_verbosity_error()

        # FP8 Dynamic Patching (Keeps RAM allocation low)
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
            for name, param in module.named_parameters(recurse=False):
                if param.dtype in fp8_types:
                    param.data = param.data.to(target_dtype)
            for child_name, child_module in module.named_children():
                patch_transformer_fp8_params(child_module, target_dtype)

        try:
            torch.set_num_threads(16)
            torch.set_num_interop_threads(16)
        except RuntimeError:
            pass

        # --- Stage 1: Text Subprocess ---
        print("\n--- [1/4] Loading Text Encoder & Encoding Prompts ---")
        if os.path.exists("tmp_embeds.pt"):
            os.remove("tmp_embeds.pt")
            
        p_proc = mp.Process(target=encode_in_subprocess, args=(MODEL_PATH, config['prompt'], config['negative_prompt']))
        p_proc.start()
        
        while p_proc.is_alive():
            p_proc.join(timeout=0.5)
            if cancel_flag:
                p_proc.terminate()
                p_proc.join()
                raise CancellationError("Cancelled during text encoding.")
        
        if not os.path.exists("tmp_embeds.pt"):
            raise RuntimeError("Text encoding failed. Check subprocess output.")
            
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = torch.load("tmp_embeds.pt")
        
        if cancel_flag: raise CancellationError("Cancelled after text encoding.")
        
        # --- Stage 2: Load FP8 Models ---
        print("--- [2/4] Loading FP8 Transformer & Enabling VRAM Protections ---")
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            MODEL_PATH,
            subfolder="transformer",
            torch_dtype=torch.float8_e4m3fn,
            local_files_only=True
        )
        patch_transformer_fp8_params(transformer, target_dtype=torch.bfloat16)
        
        if cancel_flag: raise CancellationError("Cancelled during model load.")

        pipe = LTX2Pipeline.from_pretrained(
            MODEL_PATH,
            transformer=transformer,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.set_progress_bar_config(disable=True)

        if hasattr(pipe.transformer, "set_attention_backend"):
            pipe.transformer.set_attention_backend("native")

        onload_device = torch.device("cuda")
        offload_device = torch.device("cpu")

        if hasattr(pipe.transformer, "enable_group_offload"):
            pipe.transformer.enable_group_offload(
                onload_device=onload_device, offload_device=offload_device,
                offload_type="block_level", num_blocks_per_group=2, use_stream=True
            )
        else:
            apply_group_offloading(pipe.transformer, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2)

        if hasattr(pipe, "enable_vae_slicing"): pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"): pipe.enable_vae_tiling()
        if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"): pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "tile_sample_min_size"): pipe.vae.tile_sample_min_size = 128
        if hasattr(pipe.vae, "tile_latent_min_size"): pipe.vae.tile_latent_min_size = 16

        if cancel_flag: raise CancellationError("Cancelled before generation.")

        # --- Stage 3: Generation Loop ---
        generation_start_time = time.time()
        print(f"--- [3/4] Generating Base Video ({config['width']}x{config['height']}) ---")
        
        root.after(0, progress_var.set, 0)
        
        def step_callback(pipe_instance, step_index, timestep, callback_kwargs):
            if cancel_flag:
                raise CancellationError("Cancelled by user during diffusion process.")
                
            elapsed = time.time() - generation_start_time
            print(f"  --> Completed Step {step_index + 1}/8 ({elapsed:.1f}s elapsed)")
            root.after(0, progress_var.set, step_index + 1)
                
            return callback_kwargs

        with torch.inference_mode():
            output_type = "latent" if config.get('upscale') else "np"
            output = pipe(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                width=config['width'],
                height=config['height'],
                num_frames=config['frames'],
                frame_rate=config['fps'],
                sigmas=DISTILLED_SIGMA_VALUES,
                guidance_scale=1.0,
                callback_on_step_end=step_callback,
                generator=torch.Generator("cuda").manual_seed(config['active_seed']),
                output_type=output_type,
                return_dict=False,
            )

        if cancel_flag: raise CancellationError("Cancelled before video export.")

        video = output[0]
        audio = output[1] if len(output) > 1 else None
        
        # --- Stage 4: Upscaling ---
        if config.get('upscale'):
            print(f"--- [4/4] Upscaling Latents to {config['width']*2}x{config['height']*2} ---")
            
            # Move transformer out to free VRAM for the upscaler
            pipe.transformer.to("cpu") 
            torch.cuda.empty_cache()
            
            upscale_pipe = LTXLatentUpsamplePipeline.from_pretrained(
                UPSCALER_PATH, 
                vae=pipe.vae,
                torch_dtype=torch.bfloat16, 
                local_files_only=True
            )
            # Move upscaler to GPU
            upscale_pipe.to("cuda")
            
            print("  --> Running spatial upscaler diffusion (this may take a minute)...")
            with torch.inference_mode():
                # The upscaler runs the enhancement and decodes it via the shared VAE
                video = upscale_pipe(
                    latents=video, 
                    generator=torch.Generator("cuda").manual_seed(config['active_seed']),
                    output_type="np", 
                    return_dict=False
                )[0]
                
            # Free upscaler from VRAM
            upscale_pipe.to("cpu")
            torch.cuda.empty_cache()
        else:
            print("--- [4/4] Skipping Upscaler (Native Resolution selected) ---")
            print("  --> Denoising complete. Moving VAE to GPU for decode...")
            pipe.vae.to("cuda")
            if hasattr(pipe, "audio_vae") and pipe.audio_vae is not None:
                pipe.audio_vae.to("cuda")
            if hasattr(pipe, "vocoder") and pipe.vocoder is not None:
                pipe.vocoder.to("cuda")

        print("  --> Exporting final video...")
        out_w = config['width'] * 2 if config.get('upscale') else config['width']
        out_h = config['height'] * 2 if config.get('upscale') else config['height']
        output_file = f"output_{out_w}x{out_h}_{config['frames']}f_seed{config['active_seed']}.mp4"
        sample_rate = getattr(pipe.vocoder.config, "output_sampling_rate", 24000) if hasattr(pipe, "vocoder") and pipe.vocoder else 24000

        encode_video(
            video[0], # Pass the extracted first video in the batch to the encoder
            audio=audio[0].float().cpu() if audio is not None else None,
            audio_sample_rate=sample_rate,
            output_path=output_file,
            fps=int(config['fps']),
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
        if os.path.exists("tmp_embeds.pt"):
            os.remove("tmp_embeds.pt")
        try: del pipe; del transformer
        except: pass
        try: del upscale_pipe
        except: pass
        
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        root.after(0, btn_generate.config, {"state": "normal"})
        root.after(0, btn_cancel.config, {"state": "disabled"})
        print("-" * 60)

# =========================================================================
# 5. Main Application / GUI
# =========================================================================
def main():
    def load_saved_config():
        defaults = {
            "prompt": "A high-quality cinematic shot of a classic sports car driving along a coastal highway at sunset, vibrant orange horizon, clear ocean view.",
            "negative_prompt": "", 
            "width": 768,
            "height": 512,
            "frames": 65,
            "fps": 24.0,
            "seed": "42",
            "upscale": False
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
    
    ttk.Label(main_frame, text="Positive Prompt:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
    text_prompt = tk.Text(main_frame, height=3, wrap=tk.WORD, font=("Arial", 10))
    text_prompt.pack(fill=tk.X, pady=(0, 12))
    text_prompt.insert(tk.END, config['prompt'])
    
    np_header = ttk.Frame(main_frame)
    np_header.pack(fill=tk.X)
    ttk.Label(np_header, text="Negative Prompt (Leave blank for default):", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
    
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
    res_combo['values'] = ("1280x704 (High)", "1024x576 (Medium)", "768x512 (Low)", "Custom")
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
    ttk.Checkbutton(res_frame, text="Enable 2x Latent Upscaler", variable=upscale_var).pack(pady=(5,0))
    
    def on_res_select(event):
        val = res_combo.get()
        entry_w.config(state="normal"); entry_h.config(state="normal")
        if val != "Custom":
            w, h = val.split(" ")[0].split("x")
            entry_w.delete(0, tk.END); entry_w.insert(0, w)
            entry_h.delete(0, tk.END); entry_h.insert(0, h)
            entry_w.config(state="disabled"); entry_h.config(state="disabled")
            
    res_combo.bind("<<ComboboxSelected>>", on_res_select)
    if f"{config['width']}x{config['height']}" not in [v.split(" ")[0] for v in res_combo['values'][:3]]:
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
                
            config.update({
                'prompt': p, 'negative_prompt': np_val,
                'width': w_adj, 'height': h_adj, 'fps': fps, 'frames': aligned_frames, 'seed': s_val,
                'active_seed': active_seed,
                'upscale': upscale_var.get()
            })
            save_config(config)
            
            btn_generate.config(state="disabled")
            btn_cancel.config(state="normal")
            progress_var.set(0)
            
            thread = threading.Thread(target=generation_worker, args=(config, root, progress_var, btn_generate, btn_cancel))
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
    btn_cancel.pack(side=tk.RIGHT, fill=tk.X, expand=True, ipady=8, padx=(4, 0))

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
