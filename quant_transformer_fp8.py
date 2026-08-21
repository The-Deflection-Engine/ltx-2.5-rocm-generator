#!/usr/bin/env python3
import os
import torch
from diffusers import LTX2Pipeline

ORIGINAL_MODEL_PATH = os.path.abspath("./local_ltx25_model")
NEW_MODEL_PATH = os.path.abspath("./local_ltx25_fp8")

print(f"--- [1/3] Loading original 16-bit model from {ORIGINAL_MODEL_PATH} ---")
pipe = LTX2Pipeline.from_pretrained(
    ORIGINAL_MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
)

print("--- [2/3] Converting Transformer to FP8 (8-bit)... ---")
fp8_target = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float8_e4m3fnuz

# The original, simple CPU loop from our first successful run
for name, module in pipe.transformer.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.to(fp8_target)

print(f"--- [3/3] Saving new optimized model to {NEW_MODEL_PATH} ---")
# Safetensors natively supports saving FP8 weights
pipe.save_pretrained(NEW_MODEL_PATH, safe_serialization=True)

print("SUCCESS! The model is saved. You can now run your main generation script.")
