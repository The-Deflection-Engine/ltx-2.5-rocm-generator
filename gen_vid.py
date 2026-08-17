import gc
import os
import torch
from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = os.path.abspath("./local_ltx25_model")
prompt = "A high-quality cinematic shot of a classic sports car driving along a coastal highway at sunset."
negative_prompt = DEFAULT_NEGATIVE_PROMPT

# =========================================================================
# STAGE 1: Encode Text with Gemma (~25GB RAM), then completely purge
# =========================================================================
print("--- [Stage 1/2] Encoding prompt embeddings on CPU ---")
tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/tokenizer")
text_encoder = AutoModel.from_pretrained(
    f"{MODEL_PATH}/text_encoder",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
text_encoder.eval()

with torch.no_grad():
    text_inputs = tokenizer(
        [prompt, negative_prompt],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    prompt_outputs = text_encoder(
        input_ids=text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
    )
    prompt_embeds = prompt_outputs.last_hidden_state[0:1]
    negative_prompt_embeds = prompt_outputs.last_hidden_state[1:2]
    prompt_attention_mask = text_inputs.attention_mask[0:1]
    negative_prompt_attention_mask = text_inputs.attention_mask[1:2]

print("Purging Gemma text encoder from RAM...")
del text_encoder, tokenizer, prompt_outputs, text_inputs
gc.collect()
torch.cuda.empty_cache()
print("RAM cleared successfully (Stage 1 complete).")

# =========================================================================
# STAGE 2: Load FP8 Transformer & Pipeline (text_encoder = None)
# =========================================================================
print("--- [Stage 2/2] Loading FP8 Diffusion Transformer & Pipeline ---")
fp8_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float8_e4m3fnuz

print("Loading transformer directly in FP8...")
transformer = LTX2VideoTransformer3DModel.from_pretrained(
    f"{MODEL_PATH}/transformer",
    torch_dtype=fp8_dtype,
    local_files_only=True,
)

print("Assembling pipeline (Gemma excluded to protect 32GB RAM limit)...")
pipe = LTX2Pipeline.from_pretrained(
    MODEL_PATH,
    transformer=transformer,
    text_encoder=None,
    tokenizer=None,
    dtype=torch.bfloat16,
    local_files_only=True,
)

pipe.set_progress_bar_config(disable=False)

print("Configuring sequential CPU offload for AMD RX 9070 XT...")
pipe.enable_sequential_cpu_offload()

if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
if hasattr(pipe, "enable_vae_tiling"):
    pipe.enable_vae_tiling()

# =========================================================================
# STAGE 3: Generate Video
# =========================================================================
print("Starting video generation on AMD Radeon RX 9070 XT...")
with torch.inference_mode():
    video, audio = pipe(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        width=704,
        height=480,
        num_frames=65,
        frame_rate=24.0,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(42),
        output_type="np",
    )

output_file = "output_fp8.mp4"
sample_rate = getattr(pipe.vocoder.config, "output_sampling_rate", 24000)

encode_video(
    video[0],
    audio=audio[0].float().cpu() if audio is not None else None,
    audio_sample_rate=sample_rate,
    output_path=output_file,
    fps=24,
)

print(f"Generation complete! Video saved as: {output_file}")
