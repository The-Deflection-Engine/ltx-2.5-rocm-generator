Same prompt (`prompt.txt`), rendered two ways.

## Base — `output-base.mp4`

Derived from the output filename (`output_768x512_241f_seed62805381.mp4`
before renaming) — no accompanying log was saved for this render, so the
guidance-mode fields below are inferred, not measured.

- **Resolution:** 768x512
- **Frames:** 241 (≈10.0s at the default 24fps — fps wasn't recorded, assumed default)
- **Seed:** 62805381
- **CFG quality mode / STG:** off — the filename carries no `_cfg`/`_stg` suffix.
- **2-stage upscale:** off

## Upscaled — `output-upscaled.mp4`

Confirmed directly from the GUI screenshot (`settings-upscaled.png`), not inferred.

- **Base resolution:** 768x512, 2-stage upscale → **output 1536x1024**
- **Frames:** 241
- **Seed:** 62805381 (same as the base render)
- **FPS:** 24.0
- **CFG quality mode / STG:** off
- **Render time:** 988.7s generation + VAE decode (16.6 min total, per the log tail in the screenshot)
