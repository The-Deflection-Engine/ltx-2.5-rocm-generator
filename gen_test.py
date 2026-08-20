import os

# =========================================================================
# 1. Environment & Allocator Configuration (Must precede backend imports)
# =========================================================================
# Unified memory allocator configuration for PyTorch / ROCm
os.environ["PYTORCH_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"

# ROCm / MIOpen stability flags
os.environ["MIOPEN_LOG_LEVEL"] = "3"
os.environ["MIOPEN_FIND_MODE"] = "1"
os.environ["AMD_DIRECT_DISPATCH"] = "1"
os.environ["HIP_FORCE_DEV_KERN_LAZY_COMPILE"] = "0"
os.environ["TORCH_ROCM_AOTXN_ENABLE"] = "1"
os.environ["AMD_SERIALIZE_KERNEL"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

# Multi-core CPU scheduling
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["HIP_COMPILER_NUM_THREADS"] = "16"

import gc
import time
import warnings
import torch
import torch.nn.functional as F
import diffusers
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

# Suppress deprecation and configuration warnings
warnings.filterwarnings("ignore", category=FutureWarning)
diffusers.logging.set_verbosity_error()

torch.set_num_threads(16)
torch.set_num_interop_threads(16)

# =========================================================================
# 2. Dynamic FP8 Linear Cast Hook
# =========================================================================
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

# =========================================================================
# 3. Parameters & Prompt
# =========================================================================
MODEL_PATH = os.path.abspath("./local_ltx25_model")
prompt = "A high-quality cinematic shot of a classic sports car driving along a coastal highway at sunset, vibrant orange horizon, clear ocean view."
negative_prompt = DEFAULT_NEGATIVE_PROMPT

WIDTH = 1280
HEIGHT = 704
NUM_FRAMES = 65  # 8k + 1 rule (~2.7s @ 24 FPS)
FPS = 24.0
SEED = 42

# =========================================================================
# 4. Stage 1: Load Pipeline & Encode Prompts
# =========================================================================
print("--- [1/3] Loading pipeline & encoding prompts ---")
pipe = LTX2Pipeline.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.set_progress_bar_config(disable=False)

# Offload text encoder to CPU for encoding pass
pipe.text_encoder.to("cpu")
with torch.no_grad():
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device="cpu",
    )

# Free text encoder and tokenizer from memory
pipe.register_modules(text_encoder=None, tokenizer=None)
gc.collect()
torch.cuda.empty_cache()

# =========================================================================
# 5. Stage 2: Quantize Transformer & Configure Memory Safeguards
# =========================================================================
print("--- [2/3] Quantizing Transformer & Enabling VRAM Protections ---")
fp8_target = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float8_e4m3fnuz

for name, module in pipe.transformer.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.to(fp8_target)

# Sequential offloading to prevent simultaneous layer allocations
pipe.enable_sequential_cpu_offload()

# VAE Tiling and Slicing guards: Chunk decoding to avoid monolithic kernel timeouts
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
if hasattr(pipe, "enable_vae_tiling"):
    pipe.enable_vae_tiling()
if hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()
if hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()

# Set bounded chunk sizes on the VAE if supported
if hasattr(pipe.vae, "tile_sample_min_size"):
    pipe.vae.tile_sample_min_size = 256
if hasattr(pipe.vae, "tile_latent_min_size"):
    pipe.vae.tile_latent_min_size = 32

start_time = time.time()
def step_callback(pipe_instance, step_index, timestep, callback_kwargs):
    elapsed = time.time() - start_time
    print(f"  --> Completed Step {step_index + 1}/8 ({elapsed:.1f}s elapsed)")
    return callback_kwargs

# =========================================================================
# 6. Stage 3: Generate Video
# =========================================================================
print(f"--- [3/3] Generating {WIDTH}x{HEIGHT} Video ({NUM_FRAMES} frames) on RX 9070 XT ---")
with torch.inference_mode():
    output = pipe(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        width=WIDTH,
        height=HEIGHT,
        num_frames=NUM_FRAMES,
        frame_rate=FPS,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        callback_on_step_end=step_callback,
        generator=torch.Generator("cuda").manual_seed(SEED),
        output_type="np",
        return_dict=False,
    )

video, audio = output[0], output[1] if len(output) > 1 else None

# =========================================================================
# 7. Export MP4 with Synchronized Audio
# =========================================================================
output_file = f"output_{WIDTH}x{HEIGHT}_{NUM_FRAMES}f.mp4"
sample_rate = getattr(pipe.vocoder.config, "output_sampling_rate", 24000)

encode_video(
    video[0],
    audio=audio[0].float().cpu() if audio is not None else None,
    audio_sample_rate=sample_rate,
    output_path=output_file,
    fps=int(FPS),
)

print(f"\nSUCCESS! Video saved as: {output_file}")
