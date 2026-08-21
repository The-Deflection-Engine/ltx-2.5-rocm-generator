#!/usr/bin/env python3
"""Headless LTX-2.5 generation. Settings come from ltx2_config.json; flags override.

This is a front-end, not a second implementation: it drives the same
`generation_worker` the GUI uses. That function never touches Tk -- it only
calls `.after()`, `.set()` and `.config()` on objects handed to it -- so the
stubs below stand in for the widgets and everything else is shared.

    python cli_gen_vid.py                       # run what's in the config
    python cli_gen_vid.py --prompt "a red car"  # override just the prompt
    python cli_gen_vid.py --dry-run             # resolve settings, print, exit
"""
import argparse
import json
import os
import random
import signal
import sys

import ltx_engine as gv


# --- Widget stand-ins ---------------------------------------------------------
# generation_worker marshals GUI updates through root.after(delay, fn, *args).
# Off the main thread that is required; here it is just an indirection, so run
# the callback immediately and let the worker's own print() calls be the UI.
class _Root:
    @staticmethod
    def after(_delay, fn=None, *args):
        if callable(fn):
            fn(*args)


class _Var:
    def set(self, _value):
        pass


class _Widget:
    def config(self, *_a, **_kw):
        pass


def resolve(config, args):
    """Apply CLI overrides, then the same snapping the GUI applies."""
    if args.prompt is not None:
        config["prompt"] = args.prompt
    if args.negative is not None:
        config["negative_prompt"] = args.negative
    if args.width:
        config["width"] = args.width
    if args.height:
        config["height"] = args.height
    if args.fps:
        config["fps"] = args.fps
    if args.image is not None:
        config["image_path"] = args.image
        config["mode"] = "image2video" if args.image else "text2video"
    if args.upscale is not None:
        config["upscale"] = args.upscale
    if args.cfg is not None:
        config["cfg_mode"] = args.cfg
    if args.stg is not None:
        config["stg_mode"] = args.stg
    if args.seed is not None:
        config["seed"] = args.seed

    if args.seconds is not None:
        config["frames"] = int(args.seconds * float(config["fps"]))
    elif args.frames is not None:
        config["frames"] = args.frames

    raw_w, raw_h = int(config["width"]), int(config["height"])
    raw_f = int(config["frames"])
    config["width"], config["height"] = gv.snap_dimension(raw_w), gv.snap_dimension(raw_h)
    config["frames"] = gv.align_frames(raw_f)
    if (config["width"], config["height"]) != (raw_w, raw_h):
        print(f"[!] Resolution {raw_w}x{raw_h} -> {config['width']}x{config['height']} "
              f"(multiples of {gv.SPATIAL_COMPRESSION}, min {gv.MIN_DIMENSION}).")
    if config["frames"] != raw_f:
        print(f"[!] Frames {raw_f} -> {config['frames']} (8k+1 rule).")

    seed = str(config.get("seed", "42")).strip().lower()
    config["active_seed"] = random.randint(0, 2**32 - 1) if seed == "r" else int(seed)
    return config


def summarise(config):
    up = bool(config.get("upscale"))
    w, h, f = int(config["width"]), int(config["height"]), int(config["frames"])
    out_w, out_h = (w * 2, h * 2) if up else (w, h)
    # Under Auto Duration the model picks the length, so f above is never what
    # actually runs -- size the token/VRAM estimate against the worst case it's
    # allowed to pick instead, same as the GUI's pre-flight check does.
    if config.get("auto_duration"):
        auto_max = min(float(config.get("auto_max_seconds", 5.0)), gv.AUTO_DURATION_CAP_S)
        check_frames = int(auto_max * float(config["fps"]))
    else:
        check_frames = f
    tokens = gv.latent_tokens(w, h, check_frames, up)
    eff = tokens * 2 if config.get("cfg_mode") else tokens
    est = gv.VRAM_BASE_GB + gv.VRAM_GB_PER_TOKEN * eff
    _, _, vram_total = gv.hw_monitor.get_gpu_stats()
    threshold = gv.token_warn_threshold(config)

    passes = 1 + (1 if config.get("cfg_mode") else 0) + (1 if config.get("stg_mode") else 0)
    print(f"  mode       : {config.get('mode', 'text2video')}"
          f"{' + CFG quality' if config.get('cfg_mode') else ''}"
          f"{' + STG' if config.get('stg_mode') else ''}"
          f"{f'   ({passes} passes/step)' if passes > 1 else ''}")
    print(f"  output     : {out_w}x{out_h}  ({'2-stage' if up else 'single-stage'} from {w}x{h})")
    if config.get("auto_duration"):
        print(f"  length     : auto ({config.get('auto_min_seconds', 2.0):.1f}-"
              f"{auto_max:.1f}s) -- tokens below are the worst case")
    else:
        print(f"  length     : {f} frames @ {config['fps']}fps ({f / float(config['fps']):.2f}s)")
    print(f"  seed       : {config['active_seed']}")
    print(f"  tokens     : {tokens:,}{f'  (x2 under CFG = {eff:,})' if eff != tokens else ''}")
    print(f"  est. VRAM  : ~{est:.1f}GB"
          f"{f' of {vram_total:.1f}GB' if vram_total else ''}"
          f"   [warn above {threshold:,} tokens]")
    if config.get("mode") == "image2video":
        print(f"  image      : {config.get('image_path') or '(none!)'}")
    return eff, threshold, est


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=gv.CONFIG_FILE, help="settings file (default: %(default)s)")
    ap.add_argument("--prompt")
    ap.add_argument("--negative")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--seconds", type=float, help="length in seconds (overrides --frames)")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--seed", help="integer, or 'r' for random")
    ap.add_argument("--image", help="conditioning image for image-to-video ('' for text-to-video)")
    ap.add_argument("--upscale", action=argparse.BooleanOptionalAction,
                    help="2-stage 2x latent upscale + refine")
    ap.add_argument("--cfg", action=argparse.BooleanOptionalAction,
                    help="CFG quality mode: better adherence, ~7-8x slower, ~2x VRAM")
    ap.add_argument("--stg", action=argparse.BooleanOptionalAction,
                    help="Spatio-Temporal Guidance: targets anatomy/floating objects, "
                         "no negative prompt, ~2x slower")
    ap.add_argument("--debug", action="store_true", help="per-step timing and VRAM lines")
    ap.add_argument("--dry-run", action="store_true", help="print resolved settings and exit")
    ap.add_argument("--force", action="store_true", help="proceed past the VRAM warning")
    ap.add_argument("--save", action="store_true", help="write resolved settings back to the config")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"No config at {args.config}; using built-in defaults.")
        config = {}
    else:
        with open(args.config) as f:
            config = json.load(f)
    # Fill anything absent from the same defaults the GUI uses.
    defaults = {
        "prompt": "", "negative_prompt": "", "width": 960, "height": 544,
        "frames": 121, "fps": 24.0, "seed": "42", "upscale": False,
        "mode": "text2video", "image_path": "", "auto_duration": False,
        "auto_min_seconds": 2.0, "auto_max_seconds": 5.0, "image_crf": None,
        "cfg_mode": False, "cfg_steps": 30, "cfg_scale": 3.0, "audio_cfg_scale": 7.0, "cfg_modality_scale": 1.0,
        "stg_mode": False, "stg_scale": 1.0,
        "blocks_per_group": 4, "attention_backend": "native",
    }
    defaults.update(config)
    config = resolve(defaults, args)

    if not config["prompt"].strip():
        sys.exit("Positive prompt is empty -- set it in the config or pass --prompt.")
    if config.get("mode") == "image2video" and not os.path.exists(config.get("image_path", "")):
        sys.exit(f"Image-to-video needs a valid image: {config.get('image_path')!r}")

    gv.debug_flag = args.debug
    print("\n--- Resolved settings ---")
    eff_tokens, threshold, est = summarise(config)

    if eff_tokens > threshold and not args.force:
        print(f"\n[!] {eff_tokens:,} tokens exceeds the {threshold:,} warning threshold for this "
              f"GPU (est. ~{est:.1f}GB).\n    Risk of exhausting VRAM or tripping the driver "
              f"watchdog. Re-run with --force to proceed anyway.")
        sys.exit(2)

    if args.save:
        with open(args.config, "w") as f:
            json.dump(config, f, indent=4)
        print(f"  (settings written to {args.config})")

    if args.dry_run:
        print("\nDry run -- nothing generated.")
        return

    # Ctrl-C sets the same flag the GUI's Cancel button does, so the worker
    # stops cleanly between steps instead of leaving the GPU mid-kernel.
    def on_sigint(_sig, _frame):
        if gv.cancel_flag:
            sys.exit("\nSecond interrupt -- exiting immediately.")
        gv.cancel_flag = True
        print("\n[!] Cancelling after the current step... (Ctrl-C again to force)")

    signal.signal(signal.SIGINT, on_sigint)

    print()
    gv.generation_worker(config, _Root(), _Var(), _Widget(), _Widget(), _Widget())


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
