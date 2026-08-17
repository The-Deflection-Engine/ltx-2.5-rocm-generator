import os

# =========================================================================
# 1. Environment & Hardware Configuration
# =========================================================================
os.environ["MIOPEN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["HIP_COMPILER_NUM_THREADS"] = "16"
os.environ["AMD_DIRECT_DISPATCH"] = "1"
os.environ["HIP_FORCE_DEV_KERN_LAZY_COMPILE"] = "0"
os.environ["TORCH_ROCM_AOTXN_ENABLE"] = "1"
os.environ["AMD_SERIALIZE_KERNEL"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import gc
import time
import warnings
import torch
import torch.nn.functional as F
import diffusers
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

# Suppress library deprecation and config warnings
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

MODEL_PATH = os.path.abspath("./local_ltx25_model")
prompt = "Macro extreme close-up of a vibrant chameleon perched on a jungle vine, slowly blinking and shifting its scales from emerald green to bright turquoise, morning dew droplets clinging to leaves, soft golden hour sunlight."
negative_prompt = DEFAULT_NEGATIVE_PROMPT

# =========================================================================
# 3. Stage 1: Load Pipeline & Encode Prompts
# =========================================================================
print("--- [1/3] Loading pipeline & encoding prompts ---")
pipe = LTX2Pipeline.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.set_progress_bar_config(disable=False)

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

# Cleanly deregister text encoder from pipeline state to avoid FutureWarnings
pipe.register_modules(text_encoder=None, tokenizer=None)
gc.collect()
torch.cuda.empty_cache()

# =========================================================================
# 4. Stage 2: Quantize Transformer & Configure Memory Safeguards
# =========================================================================
print("--- [2/3] Quantizing Transformer & Enabling VAE Protection ---")
fp8_target = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float8_e4m3fnuz

for name, module in pipe.transformer.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.to(fp8_target)

pipe.enable_sequential_cpu_offload()

if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
if hasattr(pipe, "enable_vae_tiling"):
    pipe.enable_vae_tiling()
if hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()
if hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()

start_time = time.time()
def step_callback(pipe_instance, step_index, timestep, callback_kwargs):
    elapsed = time.time() - start_time
    print(f"  --> Completed Step {step_index + 1}/8 ({elapsed:.1f}s elapsed)")
    return callback_kwargs

# =========================================================================
# 5. Stage 3: Generate Video
# =========================================================================
print("--- [3/3] Generating Video on RX 9070 XT ---")
with torch.inference_mode():
    output = pipe(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        width=768,
        height=512,
        num_frames=65,
        frame_rate=24.0,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        callback_on_step_end=step_callback,
        generator=torch.Generator("cuda").manual_seed(42),
        output_type="np",
        return_dict=False,
    )

video, audio = output[0], output[1] if len(output) > 1 else None

# =========================================================================
# 6. Export MP4
# =========================================================================
output_file = "output_ltx25.mp4"
sample_rate = getattr(pipe.vocoder.config, "output_sampling_rate", 24000)

encode_video(
    video[0],
    audio=audio[0].float().cpu() if audio is not None else None,
    audio_sample_rate=sample_rate,
    output_path=output_file,
    fps=24,
)

print(f"\nSUCCESS! Video saved as: {output_file}")
