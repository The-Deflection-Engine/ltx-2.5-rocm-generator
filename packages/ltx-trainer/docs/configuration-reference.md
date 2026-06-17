# Configuration Reference

The trainer uses structured Pydantic models for configuration, making it easy to customize training parameters.
This guide covers all available configuration options and their usage.

## 📋 Overview

The main configuration class is [`LtxTrainerConfig`](../src/ltx_trainer/config.py), which includes the following
sub-configurations:

- **ModelConfig**: Base model and training mode settings
- **LoraConfig**: LoRA training parameters
- **TrainingStrategyConfig**: Training strategy settings (flexible conditioning framework)
- **OptimizationConfig**: Learning rate, batch sizes, and scheduler settings
- **AccelerationConfig**: Mixed precision and quantization settings
- **DataConfig**: Data loading parameters
- **ValidationConfig**: Validation and inference settings
- **CheckpointsConfig**: Checkpoint saving frequency and retention settings
- **HubConfig**: Hugging Face Hub integration settings
- **WandbConfig**: Weights & Biases logging settings
- **FlowMatchingConfig**: Timestep sampling parameters

## 📄 Example Configuration Files

Check out our example configurations in the `configs` directory:

- 📄 [Text-to-Video LoRA](../configs/t2v_lora.yaml) - Text-to-video LoRA training
- 📄 [Image-to-Video LoRA](../configs/i2v_lora.yaml) - Image-to-video LoRA training
- 📄 [IC-LoRA Video-to-Video](../configs/v2v_ic_lora.yaml) - IC-LoRA video-to-video training
- 📄 [Audio-to-Video LoRA](../configs/a2v_lora.yaml) - Audio-to-video LoRA training
- 📄 [Video-to-Audio LoRA](../configs/v2a_lora.yaml) - Video-to-audio (Foley) LoRA training
- 📄 [Video Extension LoRA](../configs/video_extend_lora.yaml) - Video extension (forward) LoRA training
- 📄 [Video Suffix LoRA](../configs/video_suffix_lora.yaml) - Video extension (backward) LoRA training
- 📄 [Video Inpainting LoRA](../configs/video_inpainting_lora.yaml) - Video inpainting LoRA training
- 📄 [Video Outpainting LoRA](../configs/video_outpainting_lora.yaml) - Video outpainting (spatial crop) LoRA training
- 📄 [Text-to-Audio LoRA](../configs/t2a_lora.yaml) - Text-to-audio LoRA training
- 📄 [Audio Extension LoRA](../configs/audio_extend_lora.yaml) - Audio extension (forward) LoRA training
- 📄 [Audio Suffix LoRA](../configs/audio_suffix_lora.yaml) - Audio extension (backward) LoRA training
- 📄 [Audio Inpainting LoRA](../configs/audio_inpainting_lora.yaml) - Audio inpainting LoRA training
- 📄 [Audio-to-Audio IC-LoRA](../configs/a2a_ic_lora.yaml) - Audio IC-LoRA transformation training
- 📄 [AV2AV IC-LoRA](../configs/av2av_ic_lora.yaml) - Audio+video IC-LoRA transformation training
- 📄 [T2V LoRA (Low VRAM)](../configs/t2v_lora_low_vram.yaml) - Memory-optimized config for 32GB GPUs

## ⚙️ Configuration Sections

> [!NOTE]
> The YAML snippets below show **recommended starting values**, not necessarily the code defaults.
> Fields you omit from your config file will use the code defaults from [`config.py`](../src/ltx_trainer/config.py).

### ModelConfig

Controls the base model and training mode settings.

```yaml
model:
  model_path: "/path/to/ltx-2-model.safetensors"  # Local path to model checkpoint
  text_encoder_path: "/path/to/gemma-model"       # Path to Gemma text encoder directory
  training_mode: "lora"                           # "lora" or "full"
  load_checkpoint: null                           # Path to checkpoint to resume from
```

**Key parameters:**

| Parameter           | Description                                                                                                                                                    |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_path`        | **Required.** Local path to the LTX-2 model checkpoint (`.safetensors` file). URLs are not supported.                                                          |
| `text_encoder_path` | **Required.** Path to the Gemma text encoder model directory. Download from [HuggingFace](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/). |
| `training_mode`     | Training approach - `"lora"` for LoRA training or `"full"` for full-rank fine-tuning.                                                                          |
| `load_checkpoint`   | Optional path to resume training from a checkpoint file or directory.                                                                                          |

> [!NOTE]
> LTX-2 requires both a model checkpoint and a Gemma text encoder. Both must be local paths.

### LoraConfig

LoRA-specific fine-tuning parameters (only used when `training_mode: "lora"`).

```yaml
lora:
  rank: 32         # LoRA rank (higher = more parameters)
  alpha: 32        # LoRA alpha scaling factor
  dropout: 0.0     # Dropout probability (0.0-1.0)
  target_modules: # Modules to apply LoRA to
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"
```

**Key parameters:**

| Parameter        | Description                                                                     |
|------------------|---------------------------------------------------------------------------------|
| `rank`           | LoRA rank - higher values mean more trainable parameters (typical range: 8-128) |
| `alpha`          | Alpha scaling factor - typically set equal to rank                              |
| `dropout`        | Dropout probability for regularization                                          |
| `target_modules` | List of transformer modules to apply LoRA adapters to (see below)               |

#### Understanding Target Modules

The LTX-2 transformer has separate attention and feed-forward blocks for video and audio, as well as cross-attention
modules that enable the two modalities to exchange information. Choosing the right `target_modules` is critical for
achieving good results, especially when training with audio.

**Video-only modules:**

| Module Pattern                                             | Description                     |
|------------------------------------------------------------|---------------------------------|
| `attn1.to_k`, `attn1.to_q`, `attn1.to_v`, `attn1.to_out.0` | Video self-attention            |
| `attn2.to_k`, `attn2.to_q`, `attn2.to_v`, `attn2.to_out.0` | Video cross-attention (to text) |
| `ff.net.0.proj`, `ff.net.2`                                | Video feed-forward network      |

**Audio-only modules:**

| Module Pattern                                                                     | Description                     |
|------------------------------------------------------------------------------------|---------------------------------|
| `audio_attn1.to_k`, `audio_attn1.to_q`, `audio_attn1.to_v`, `audio_attn1.to_out.0` | Audio self-attention            |
| `audio_attn2.to_k`, `audio_attn2.to_q`, `audio_attn2.to_v`, `audio_attn2.to_out.0` | Audio cross-attention (to text) |
| `audio_ff.net.0.proj`, `audio_ff.net.2`                                            | Audio feed-forward network      |

**Audio-video cross-attention modules:**

These modules enable bidirectional information flow between the audio and video modalities:

| Module Pattern                                                                                                     | Description                                           |
|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `audio_to_video_attn.to_k`, `audio_to_video_attn.to_q`, `audio_to_video_attn.to_v`, `audio_to_video_attn.to_out.0` | Video attends to audio (Q from video, K/V from audio) |
| `video_to_audio_attn.to_k`, `video_to_audio_attn.to_q`, `video_to_audio_attn.to_v`, `video_to_audio_attn.to_out.0` | Audio attends to video (Q from audio, K/V from video) |

**Recommended configurations:**

For **video-only training**, target the video attention layers:

```yaml
target_modules:
  - "attn1.to_k"
  - "attn1.to_q"
  - "attn1.to_v"
  - "attn1.to_out.0"
  - "attn2.to_k"
  - "attn2.to_q"
  - "attn2.to_v"
  - "attn2.to_out.0"
```

For **audio-video training**, use patterns that match both branches:

```yaml
target_modules:
  - "to_k"
  - "to_q"
  - "to_v"
  - "to_out.0"
```

> [!NOTE]
> Using shorter patterns like `"to_k"` will match all attention modules including `attn1.to_k`, `audio_attn1.to_k`,
> `audio_to_video_attn.to_k`, and `video_to_audio_attn.to_k`, effectively training video, audio, and cross-modal
> attention branches together.

> [!TIP]
> You can also target the feed-forward (FFN) modules (`ff.net.0.proj`, `ff.net.2` for video,
> `audio_ff.net.0.proj`, `audio_ff.net.2` for audio) to increase the LoRA's capacity and potentially
> help it capture the target distribution better.

### TrainingStrategyConfig

Configures the training strategy. The recommended strategy is `"flexible"`, which supports all conditioning scenarios through configuration.

#### Flexible Strategy

The flexible strategy provides a unified conditioning framework. Each modality (video, audio) is configured
independently with its own latents directory, generation flag, and list of conditions.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true                 # Video is denoised during training
    latents_dir: "latents"             # Directory containing precomputed video latents
    conditions:
      - type: first_frame              # Use first frame as conditioning
        probability: 0.5               # Apply this condition 50% of the time
  audio:
    is_generated: true                 # Audio is denoised during training
    latents_dir: "audio_latents"       # Directory containing precomputed audio latents
    conditions: []                     # No additional audio conditions (text-only)
```

**ModalityConfig parameters:**

| Parameter      | Description                                                                                                      |
|----------------|------------------------------------------------------------------------------------------------------------------|
| `is_generated` | `true` = modality is denoised (contributes to loss). `false` = frozen conditioning (sigma=0, no loss).           |
| `latents_dir`  | Directory name within `preprocessed_data_root` containing precomputed latents for this modality.                 |
| `conditions`   | List of conditioning configs applied during training (see condition types below). Text conditioning is implicit. |

**Condition types:**

| Type           | Parameters                                          | Description                                                                           |
|----------------|-----------------------------------------------------|---------------------------------------------------------------------------------------|
| `first_frame`  | `probability`                                       | First latent frame is clean, excluded from loss. **Video only.**                      |
| `prefix`       | `temporal_boundary`, `probability`                  | First N latent temporal units are clean. For extension forward.                       |
| `suffix`       | `temporal_boundary`, `probability`                  | Last N latent temporal units are clean. For extension backward.                       |
| `spatial_crop` | `spatial_region` (y1, x1, y2, x2 in px), `probability` | Rectangular region is clean, excluded from loss. For outpainting. **Video only.** |
| `mask`         | `mask_dir`, `probability`                           | Per-sample mask directory. Masks are thresholded at `0.5`; `1` means conditioning, `0` means generate. |
| `reference`    | `latents_dir`, `probability`                        | IC-LoRA style concatenation. Reference tokens are prepended, clean (timestep=0), no loss. |

> [!NOTE]
> The `prefix`, `suffix`, `mask`, and `reference` condition types work on both video and audio modalities —
> place them in the `video.conditions` or `audio.conditions` list as appropriate.
> `first_frame` and `spatial_crop` are video-only conditions.

> [!NOTE]
> Training conditions reference **directories** of precomputed data (within `preprocessed_data_root`),
> while validation conditions reference **individual files** (images, videos, masks) that are encoded
> on-the-fly during validation. The condition `type` names are the same, but the fields differ.

> [!NOTE]
> The legacy `text_to_video` and `video_to_video` strategies are deprecated but remain forward-compatible.
> New configs should use `name: "flexible"`.

### OptimizationConfig

Training optimization parameters including learning rates, batch sizes, and schedulers.

```yaml
optimization:
  learning_rate: 1e-4                  # Learning rate
  steps: 2000                          # Total training steps
  batch_size: 1                        # Batch size per GPU
  gradient_accumulation_steps: 1       # Steps to accumulate gradients
  max_grad_norm: 1.0                   # Gradient clipping threshold
  optimizer_type: "adamw"              # "adamw" or "adamw8bit"
  scheduler_type: "linear"             # Scheduler type
  scheduler_params: { }                # Additional scheduler parameters
  enable_gradient_checkpointing: true  # Memory optimization
```

**Key parameters:**

| Parameter                       | Description                                                                                  |
|---------------------------------|----------------------------------------------------------------------------------------------|
| `learning_rate`                 | Learning rate for optimization (typical range: 1e-5 to 1e-3)                                 |
| `steps`                         | Total number of training steps                                                               |
| `batch_size`                    | Batch size per GPU (reduce if running out of memory)                                         |
| `gradient_accumulation_steps`   | Accumulate gradients over multiple steps                                                     |
| `scheduler_type`                | LR scheduler: `"constant"`, `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"polynomial"`, `"step"` |
| `enable_gradient_checkpointing` | Trade training speed for GPU memory savings (recommended for large models)                   |

### AccelerationConfig

Hardware acceleration and compute optimization settings.

```yaml
acceleration:
  mixed_precision_mode: "bf16"                  # "no", "fp16", or "bf16"
  quantization: null                            # Quantization options
  load_text_encoder_in_8bit: false              # Load text encoder in 8-bit
  offload_optimizer_during_validation: false    # Offload optimizer state to CPU during validation
```

**Key parameters:**

| Parameter                             | Description                                                                                                                                                                              |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mixed_precision_mode`                | Precision mode - `"bf16"` recommended for modern GPUs                                                                                                                                    |
| `quantization`                        | Model quantization: `null`, `"int8-quanto"`, `"int4-quanto"`, `"int2-quanto"`, `"fp8-quanto"`, or `"fp8uz-quanto"`                                                                       |
| `load_text_encoder_in_8bit`           | Load the Gemma text encoder in 8-bit to save GPU memory                                                                                                                                  |
| `offload_optimizer_during_validation` | Move optimizer state to CPU before validation video sampling and back afterwards. Useful when validation OOMs because VAE decoder + transformer + optimizer state can't coexist on the GPU (full fine-tune, high-rank LoRA). No effect for FSDP. |

### DataConfig

Data loading and processing configuration.

```yaml
data:
  preprocessed_data_root: "/path/to/preprocessed/data"  # Path to precomputed dataset
  num_dataloader_workers: 2                             # Background data loading workers
```

**Key parameters:**

| Parameter                | Description                                                                                |
|--------------------------|--------------------------------------------------------------------------------------------|
| `preprocessed_data_root` | Path to your preprocessed dataset directory produced by `process_dataset.py` (contains `latents/`, `conditions/`, etc.) |
| `num_dataloader_workers` | Number of parallel data loading processes (0 = synchronous loading, useful when debugging) |

### ValidationConfig

Validation and inference settings for monitoring training progress. Validation samples use a self-describing
format where each sample specifies its own prompt and conditions.

```yaml
validation:
  samples:
    - prompt: "A cat playing with a ball"
      conditions:
        - type: first_frame
          image_or_video: "/path/to/image.png"
    - prompt: "A dog running in a field"
  video_dims: [576, 576, 89]                # Output dimensions: [width, height, frames]
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"  # Negative prompt for all samples
  frame_rate: 25.0                          # Output video frame rate (fps)
  seed: 42                                  # Random seed for reproducibility
  inference_steps: 30                       # Number of denoising steps
  interval: 100                             # Run validation every N steps (null to disable)
  guidance_scale: 4.0                       # CFG scale (higher = stronger prompt adherence)
  stg_scale: 1.0                            # STG scale (0.0 to disable)
  stg_blocks: [29]                          # Transformer blocks to apply STG perturbation
  stg_mode: "stg_av"                        # STG mode: "stg_av" (audio+video) or "stg_v" (video only)
  generate_audio: true                      # Whether to generate audio during validation
  generate_video: true                      # Whether to generate video during validation
  skip_initial_validation: false            # Skip validation at step 0
```

**Key parameters:**

| Parameter                | Description                                                                                                              |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `samples`                | List of `ValidationSample` objects (see below). Replaces the legacy `prompts`/`images`/`reference_videos` fields.        |
| `video_dims`             | Output dimensions `[width, height, frames]`. Width/height must be divisible by 32, frames must satisfy `frames % 8 == 1` |
| `interval`               | Steps between validation runs (set to `null` to disable)                                                                 |
| `guidance_scale`         | CFG (Classifier-Free Guidance) scale. Recommended: 4.0                                                                   |
| `stg_scale`              | STG (Spatio-Temporal Guidance) scale. 0.0 disables STG. Recommended: 1.0                                                 |
| `stg_blocks`             | Transformer blocks to perturb for STG. Recommended: `[29]` (single block)                                                |
| `stg_mode`               | STG mode: `"stg_av"` perturbs both audio and video, `"stg_v"` perturbs video only                                        |
| `generate_audio`         | Whether to generate audio in validation samples                                                                          |
| `generate_video`         | Whether to generate video in validation samples. Set to `false` for V2A (video-to-audio) validation. Default: `true`     |
| `skip_initial_validation`| Skip validation video sampling at step 0 (beginning of training)                                                         |

#### ValidationSample

Each sample in the `samples` list has:

| Field        | Description                                                                                     |
|--------------|-------------------------------------------------------------------------------------------------|
| `prompt`     | Text prompt for this validation sample.                                                         |
| `conditions` | List of validation conditions (see types below). Empty list = text-only generation.             |
| `video_dims` | Optional per-sample override for `(width, height, frames)`. Inherits from `ValidationConfig` if not set. |
| `seed`       | Optional per-sample override for random seed. Inherits from `ValidationConfig` if not set.      |

#### Validation Condition Types

| Type             | Parameters                                                 | Description                                                             |
|------------------|------------------------------------------------------------|-------------------------------------------------------------------------|
| `first_frame`    | `image_or_video` (path)                                    | Use the first frame of the image/video as conditioning.                 |
| `prefix`         | `video` or `audio` (path), optional `num_frames`/`duration`| Use a video/audio clip as temporal prefix (for extension forward).      |
| `suffix`         | `video` or `audio` (path), optional `num_frames`/`duration`| Use a video/audio clip as temporal suffix (for extension backward).     |
| `spatial_crop`   | `video` (path), `spatial_region` (y1, x1, y2, x2)         | Provide spatial context for outpainting. Video only.                    |
| `mask`           | `video` or `audio` (path), `mask` (path)                   | Mask-based inpainting with a binary mask file.                          |
| `reference`      | `video` or `audio` (path), optional video-reference `downscale_factor`, `temporal_scale_factor`, `include_in_output` | IC-LoRA style reference conditioning. |
| `video_to_audio` | `video` (path)                                             | Freeze video, generate audio. For Foley/V2A tasks.                      |
| `audio_to_video` | `audio` (path)                                             | Freeze audio, generate video. For audio-driven generation.              |

For video `reference` validation conditions, `downscale_factor` is the spatial reference scale and
`temporal_scale_factor` is the temporal reference scale. Set both to match the factors used when
preprocessing video reference latents for training; validation media is encoded on the fly and cannot infer
those factors from the training dataset.

> [!NOTE]
> The legacy fields `prompts`, `images`, and `reference_videos` are deprecated but auto-converted to `samples`
> internally. New configs should use the `samples` format.

### CheckpointsConfig

Model checkpointing configuration.

```yaml
checkpoints:
  interval: 250       # Steps between checkpoint saves (null = disabled)
  keep_last_n: 3      # Number of recent checkpoints to retain
  precision: bfloat16 # Precision for saved weights (bfloat16 or float32)
  no_resume: false            # Ignore saved state, start from step 0
  save_training_state: "minimal"  # "full", "minimal", or "off"
```

**Key parameters:**

| Parameter     | Description                                                                   |
|---------------|-------------------------------------------------------------------------------|
| `interval`    | Steps between intermediate checkpoint saves (set to `null` to disable)        |
| `keep_last_n` | Number of most recent checkpoints to keep (-1 = keep all)                     |
| `precision`   | Precision for saved checkpoint weights: `"bfloat16"` (default) or `"float32"` |
| `no_resume` | When `true`, ignore saved training state and start from step 0. Model weights from `load_checkpoint` are still loaded. |
| `save_training_state` | Save training state for resume: `"full"` (optimizer + scheduler + RNG), `"minimal"` (scheduler + RNG only, sufficient for LoRA), `"off"` (no resume). |

### HubConfig

Hugging Face Hub integration for automatic model uploads.

```yaml
hub:
  push_to_hub: false                   # Enable Hub uploading
  hub_model_id: "username/model-name"  # Hub repository ID
```

**Key parameters:**

| Parameter      | Description                                                      |
|----------------|------------------------------------------------------------------|
| `push_to_hub`  | Whether to automatically push trained models to Hugging Face Hub |
| `hub_model_id` | Repository ID in format `"username/repository-name"`             |

### WandbConfig

Weights & Biases logging configuration.

```yaml
wandb:
  enabled: false               # Enable W&B logging
  project: "ltx-2-trainer"     # W&B project name
  entity: null                 # W&B username or team
  tags: [ ]                    # Tags for the run
  log_validation_videos: true  # Log validation videos to W&B
```

**Key parameters:**

| Parameter               | Description                                      |
|-------------------------|--------------------------------------------------|
| `enabled`               | Whether to enable W&B logging                    |
| `project`               | W&B project name                                 |
| `entity`                | W&B username or team (null uses default account) |
| `log_validation_videos` | Whether to log validation videos to W&B          |

### FlowMatchingConfig

Flow matching training configuration for timestep sampling.

```yaml
flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"  # Timestep sampling strategy
  timestep_sampling_params: { }                   # Additional sampling parameters
```

**Key parameters:**

| Parameter                  | Description                                                |
|----------------------------|------------------------------------------------------------|
| `timestep_sampling_mode`   | Sampling strategy: `"uniform"` or `"shifted_logit_normal"` |
| `timestep_sampling_params` | Additional parameters for the sampling strategy            |

### General Configuration

Top-level settings for the training run.

```yaml
seed: 42                                  # Random seed for reproducibility
output_dir: "outputs/my_training_run"     # Directory for outputs (checkpoints, validation videos, logs)
```

| Parameter    | Description                                              |
|--------------|----------------------------------------------------------|
| `seed`       | Random seed for reproducibility (default: `42`)          |
| `output_dir` | Directory to save outputs (default: `"outputs"`)         |

## 🚀 Next Steps

Once you've configured your training parameters:

- Set up your dataset using [Dataset Preparation](dataset-preparation.md)
- Choose your training approach in [Training Modes](training-modes.md)
- Start training with the [Training Guide](training-guide.md)
