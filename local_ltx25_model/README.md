# Patched `model_index.json`

`model_index.json` in this directory has been manually patched and differs from the stock file distributed with the model repo.

## What changed

`text_encoder` was changed to:

```json
"text_encoder": [
    "transformers",
    "Gemma4UnifiedForConditionalGeneration"
]
```

## Why

`LTX2Pipeline.__init__` (`diffusers/pipelines/ltx2/pipeline_ltx2.py`) types `text_encoder` as `Gemma3ForConditionalGeneration | Gemma4UnifiedForConditionalGeneration` — two valid classes. This checkpoint's actual text encoder weights are the newer **Gemma4Unified** variant, not Gemma3. The stock `model_index.json` pointed at the wrong class, so `from_pretrained` tried to load Gemma4Unified weights into a Gemma3 model and failed. Patching the class name in the index to match the actual weights fixed loading.

## If you re-download the model

A fresh download from the upstream repo may reintroduce the original (wrong) class reference. Check `text_encoder` in `model_index.json` and re-patch it to `Gemma4UnifiedForConditionalGeneration` if loading fails with a class/shape mismatch on the text encoder.
