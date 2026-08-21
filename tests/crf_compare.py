#!/usr/bin/env python3
"""How much detail does the image-conditioning CRF throw away?

Answers review item #2 without generating anything. LTX-2.5 re-compresses the
conditioning frame through H.264 before encoding it, at a CRF that depends on
the checkpoint generation: 18 for LTX-2.5, 33 for 2.3 and earlier.

Because `generate_video.py` builds the pipeline with `text_encoder=None` (to
avoid loading 23GB it does not need), diffusers' auto-detect
(`utils.resolve_default_image_crf`) cannot see a Gemma-4 encoder and falls
through to 33. So i2v conditioning is currently compressed far harder than the
checkpoint expects.

This round-trips a real image through both CRFs and measures the difference,
so the decision rests on a number rather than an assumption.

    python tests/crf_compare.py path/to/image.png
    python tests/crf_compare.py image.png --crfs 0 18 33 40
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def roundtrip(src, crf, tmpdir):
    """Encode a still through H.264 at `crf` and decode it back, as the
    pipeline does internally. Returns the decoded image as a float array."""
    mp4 = Path(tmpdir) / f"c{crf}.mp4"
    png = Path(tmpdir) / f"c{crf}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(src),
                    "-c:v", "libx264", "-crf", str(crf), "-t", "0.1",
                    "-pix_fmt", "yuv420p", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4),
                    "-frames:v", "1", str(png)], check=True)
    return np.asarray(Image.open(png).convert("RGB"), dtype=np.float64), mp4.stat().st_size


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--crfs", type=int, nargs="+", default=[18, 33],
                    help="CRFs to compare (default: the LTX-2.5 and 2.3 values)")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found.")
    src = Path(args.image)
    if not src.exists():
        sys.exit(f"No such image: {src}")

    original = np.asarray(Image.open(src).convert("RGB"), dtype=np.float64)
    print(f"source: {src.name}  {original.shape[1]}x{original.shape[0]}\n")
    print(f"{'CRF':>5} {'PSNR vs source':>16} {'mean abs err':>14} {'max err':>9} {'size':>10}")

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for crf in args.crfs:
            decoded, nbytes = roundtrip(src, crf, tmp)
            if decoded.shape != original.shape:
                print(f"{crf:>5}  (skipped: ffmpeg returned {decoded.shape}, "
                      f"expected {original.shape} -- odd dimensions get padded)")
                continue
            d = np.abs(decoded - original)
            results[crf] = psnr(original, decoded)
            print(f"{crf:>5} {results[crf]:>15.2f}dB {d.mean():>14.2f} "
                  f"{d.max():>9.0f} {nbytes/1024:>9.0f}K")

    if len(results) >= 2:
        lo, hi = min(results), max(results)
        print(f"\nCRF {lo} preserves {results[lo] - results[hi]:.2f}dB more than CRF {hi}.")
        print("Higher PSNR = closer to the source. ~1dB is marginal; >3dB is a")
        print("difference you can see in fine texture and edges.")
        print(f"\nThe pipeline uses CRF 33 today when text_encoder=None; LTX-2.5")
        print("was trained at 18. See review item #2.")


if __name__ == "__main__":
    main()
