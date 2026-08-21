#!/usr/bin/env python3
"""Time the LTX-2.5 VAE decode at several tile sizes.

VAE decode is ~2/3 of wall-clock on a 2-stage run, and tile geometry is the main
knob on it: bigger tiles mean fewer overlap-blend passes, at the cost of VRAM and
ring-timeout headroom. This decodes a latent of the real post-upscale shape and
reports time + peak VRAM per configuration.

    python bench_vae_tiles.py [--frames N] [--configs 512,768,1024]
"""
import argparse
import os
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.9")
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import torch
from diffusers import AutoencoderKLLTX2Video

MODEL_PATH = os.path.abspath("./local_ltx25_fp8")
GB = 1024 ** 3
SPATIAL_COMPRESSION = 32   # LTX-2.5 VAE: 32px -> 1 latent px
TEMPORAL_COMPRESSION = 8   #              8 frames -> 1 latent frame


def main():
    ap = argparse.ArgumentParser()
    # 13 latent frames == 97 output frames, the shape from the logged 2-stage run.
    ap.add_argument("--latent-frames", type=int, default=13)
    ap.add_argument("--latent-h", type=int, default=32)   # 32 * 32 = 1024px
    ap.add_argument("--latent-w", type=int, default=48)   # 48 * 32 = 1536px
    ap.add_argument("--configs", default="512,768,1024",
                    help="comma-separated tile sizes; stride is 87.5%% of each")
    ap.add_argument("--tile-frames", default="24",
                    help="comma-separated temporal tile sizes (stride is 2/3 of each). "
                         "Below this many frames temporal tiling never engages, so a "
                         "short clip silently skips the blending this measures.")
    args = ap.parse_args()

    h_px, w_px = args.latent_h * 32, args.latent_w * 32
    frames = (args.latent_frames - 1) * 8 + 1
    print(f"Decoding {w_px}x{h_px}, {frames} frames "
          f"(latent {args.latent_frames}x{args.latent_h}x{args.latent_w})\n")

    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True,
    ).eval().to("cuda")
    vae.use_framewise_decoding = True
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()

    ch = vae.config.latent_channels
    latents = torch.randn(1, ch, args.latent_frames, args.latent_h, args.latent_w,
                          dtype=torch.bfloat16, device="cuda")

    results = []
    combos = [(int(t), int(f))
              for f in args.tile_frames.split(",")
              for t in args.configs.split(",")]

    print(f"{'spatial':>9} {'temporal':>9} {'grid':>10} {'time':>9} {'peak':>8}", flush=True)
    for tile, tframes in combos:
        # Strides MUST land on a latent boundary: spatial compression is 32,
        # temporal is 8. A ragged stride (e.g. 384px -> 336 == 10.5 latent px)
        # is punished savagely -- it was 3x slower than a 512px tile despite
        # decoding fewer tiles, which looked like a real result until the
        # arithmetic explained it. Snap both strides down to a whole boundary.
        stride = max(SPATIAL_COMPRESSION, (int(tile * 0.875) // SPATIAL_COMPRESSION) * SPATIAL_COMPRESSION)
        tstride = max(TEMPORAL_COMPRESSION,
                      (int(tframes * 2 / 3) // TEMPORAL_COMPRESSION) * TEMPORAL_COMPRESSION)
        if tile % SPATIAL_COMPRESSION or tframes % TEMPORAL_COMPRESSION:
            print(f"{tile:>7}px {tframes:>7}f  SKIPPED (tile not a multiple of "
                  f"{SPATIAL_COMPRESSION}px / {TEMPORAL_COMPRESSION}f)", flush=True)
            continue
        vae.enable_tiling(
            tile_sample_min_height=tile,
            tile_sample_min_width=tile,
            tile_sample_min_num_frames=tframes,
            tile_sample_stride_height=stride,
            tile_sample_stride_width=stride,
            tile_sample_stride_num_frames=tstride,
        )
        tiles_x = -(-(w_px - tile) // stride) + 1 if w_px > tile else 1
        tiles_y = -(-(h_px - tile) // stride) + 1 if h_px > tile else 1
        tiles_t = -(-(frames - tframes) // tstride) + 1 if frames > tframes else 1

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            with torch.no_grad():
                out = vae.decode(latents, return_dict=False)[0]
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_reserved() / GB
            print(f"{tile:>7}px {tframes:>7}f {tiles_x}x{tiles_y}x{tiles_t:<6} "
                  f"{dt:7.1f}s {peak:6.2f}GB   ({tiles_x*tiles_y*tiles_t} tile decodes)",
                  flush=True)
            results.append((tile, tframes, dt, peak))
            del out
        except torch.OutOfMemoryError:
            print(f"{tile:>7}px {tframes:>7}f  OOM", flush=True)
        except Exception as exc:
            print(f"{tile:>7}px {tframes:>7}f  FAILED: {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.empty_cache()

    if len(results) > 1:
        b_tile, b_frames, b_dt, _ = results[0]
        print(f"\nvs {b_tile}px/{b_frames}f baseline ({b_dt:.1f}s):", flush=True)
        for tile, tframes, dt, peak in results[1:]:
            print(f"  {tile:>4}px/{tframes:>3}f: {b_dt / dt:5.2f}x  "
                  f"({b_dt - dt:+7.1f}s)  peak {peak:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
