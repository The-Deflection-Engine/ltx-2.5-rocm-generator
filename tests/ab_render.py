#!/usr/bin/env python3
"""Run a set of generation variants at a fixed seed and collect the results.

Turns "spend an afternoon eyeballing renders" into "kick it off, come back".
Each variant is a config overlay, so comparing CRF 18 vs 33, or STG on vs off,
is a JSON edit rather than a code change.

    python tests/ab_render.py tests/variants.example.json --seed 3620598670

What it measures objectively: wall-clock, peak VRAM, whether it OOMed, output
path. What it does NOT measure: whether the video looks better. It extracts a
contact sheet and individual frames for that, but the judgement stays human.

Nothing here is a substitute for watching the clip -- it is a way of making sure
the comparison is actually controlled, which a manual A/B usually isn't.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_variant(name, cfg, seed, outdir, extra_args):
    """Run one variant through cli_gen_vid.py and return what happened."""
    cfg = dict(cfg)
    cfg["seed"] = str(seed)                    # fixed seed is the whole point
    fd, cfg_path = tempfile.mkstemp(prefix=f"ab_{name}_", suffix=".json")
    os.close(fd)
    Path(cfg_path).write_text(json.dumps(cfg, indent=2))

    cmd = [sys.executable, "cli_gen_vid.py", "--config", cfg_path,
           "--debug", "--force", *extra_args]
    print(f"\n=== {name} ===\n  {' '.join(cmd)}", flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    elapsed = time.time() - t0
    os.unlink(cfg_path)

    log = proc.stdout + proc.stderr
    (outdir / f"{name}.log").write_text(log)

    result = {"name": name, "elapsed": elapsed, "exit": proc.returncode,
              "config": cfg, "video": None, "peak_vram": None,
              "decode_s": None, "oom": "OutOfMemoryError" in log}

    m = re.search(r"SUCCESS! Video saved as: (\S+)", log)
    if m:
        src = REPO / m.group(1)
        if src.exists():
            dst = outdir / f"{name}_{src.name}"
            shutil.move(str(src), dst)          # move, so variants can't collide
            result["video"] = dst
    m = re.search(r"peak VRAM this run ([\d.]+)GB reserved", log)
    if m:
        result["peak_vram"] = float(m.group(1))
    m = re.search(r"VAE decode took ([\d.]+)s", log)
    if m:
        result["decode_s"] = float(m.group(1))

    dry = "--dry-run" in extra_args
    if dry:
        status = "dry-run ok" if proc.returncode == 0 else f"dry-run failed ({proc.returncode})"
    else:
        status = "ok" if result["video"] else ("OOM" if result["oom"] else f"failed ({proc.returncode})")
    print(f"  -> {status}, {elapsed:.1f}s", flush=True)
    return result


def extract_frames(video, outdir, name):
    """Contact sheet for continuity, plus first and middle frame at full size."""
    if not video or not shutil.which("ffmpeg"):
        return {}
    frames = {}
    sheet = outdir / f"{name}_sheet.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                    r"select='not(mod(n\,16))',scale=256:-1,tile=3x3",
                    "-frames:v", "1", str(sheet)], check=False)
    if sheet.exists():
        frames["sheet"] = sheet
    for label, expr in (("first", "eq(n\\,0)"), ("mid", "eq(n\\,72)")):
        f = outdir / f"{name}_{label}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
                        "-vf", f"select='{expr}'", "-frames:v", "1", str(f)],
                       check=False)
        if f.exists():
            frames[label] = f
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", help="JSON file: {base: {...}, variants: {name: {overrides}}}")
    ap.add_argument("--seed", type=int, required=True, help="fixed for every variant")
    ap.add_argument("--outdir", default=None, help="default: runs/<timestamp>")
    ap.add_argument("--only", nargs="*", help="run only these variant names")
    ap.add_argument("--dry-run", action="store_true", help="resolve and print, generate nothing")
    args, extra = ap.parse_known_args()

    spec = json.loads(Path(args.variants).read_text())
    base, variants = spec.get("base", {}), spec["variants"]
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only}
    if not variants:
        sys.exit("No variants selected.")

    outdir = Path(args.outdir or REPO / "runs" / time.strftime("%Y-%m-%d_%H%M"))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"seed {args.seed}, {len(variants)} variants -> {outdir}")

    if args.dry_run:
        extra = extra + ["--dry-run"]

    results = []
    for name, overrides in variants.items():
        cfg = {**base, **overrides}
        r = run_variant(name, cfg, args.seed, outdir, extra)
        r["frames"] = extract_frames(r["video"], outdir, name)
        results.append(r)

    # --- report -----------------------------------------------------------
    lines = [f"# A/B render — seed {args.seed}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} · {len(results)} variants", "",
             "| variant | result | time | peak VRAM | VAE decode |",
             "|---|---|---|---|---|"]
    for r in results:
        status = "ok" if r["video"] else ("**OOM**" if r["oom"] else f"**failed {r['exit']}**")
        vram = f"{r['peak_vram']:.2f}GB" if r["peak_vram"] else "—"
        decode = f"{r['decode_s']:.0f}s" if r["decode_s"] else "—"
        lines.append(f"| `{r['name']}` | {status} | {r['elapsed']:.0f}s | {vram} | {decode} |")

    lines += ["", "## What changed per variant", ""]
    for r in results:
        diff = {k: v for k, v in r["config"].items()
                if k not in base or base.get(k) != v}
        diff.pop("seed", None)
        lines.append(f"- **{r['name']}** — `{json.dumps(diff) if diff else 'baseline'}`")

    lines += ["", "## Frames", "",
              "_Objective numbers are above. These are for the judgement the "
              "harness cannot make._", ""]
    for r in results:
        if r["frames"]:
            lines.append(f"### {r['name']}")
            for label, path in r["frames"].items():
                lines.append(f"- {label}: `{path.name}`")
            lines.append("")

    report = outdir / "results.md"
    report.write_text("\n".join(lines))
    print(f"\n{'='*60}\n{report}\n")
    print("\n".join(lines[3:10]))


if __name__ == "__main__":
    main()
