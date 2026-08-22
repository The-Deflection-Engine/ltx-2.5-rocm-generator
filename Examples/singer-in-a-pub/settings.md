Derived from the output filename (`output_768x512_241f_seed62805381.mp4`
before renaming) — no accompanying log was saved for this render, so the
guidance-mode fields below are inferred, not measured.

- **Resolution:** 768x512
- **Frames:** 241 (≈10.0s at the default 24fps — fps wasn't recorded, assumed default)
- **Seed:** 62805381
- **CFG quality mode / STG:** off — the filename carries no `_cfg`/`_stg` suffix,
  and the project's convention (see the main README) is to encode guidance
  mode in the filename whenever either is on, so their absence means the
  distilled 8-step schedule, guidance-free.
- **2-stage upscale:** not recorded — can't be told apart from a single-stage
  768x512 render using the output file alone.
