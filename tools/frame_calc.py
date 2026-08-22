#!/usr/bin/env python3
import argparse
import sys

def get_ltx_frames_from_seconds(seconds: float, fps: float = 24.0) -> tuple[int, int, float]:
    """Calculate the closest 8k + 1 frame count from duration in seconds."""
    raw_frames = seconds * fps
    k = max(1, round((raw_frames - 1) / 8))
    frames = (8 * k) + 1
    actual_seconds = frames / fps
    return frames, k, actual_seconds

def get_ltx_frames_from_raw_frames(raw_frames: int, fps: float = 24.0) -> tuple[int, int, float]:
    """Snap an arbitrary frame count to the nearest valid 8k + 1 count."""
    k = max(1, round((raw_frames - 1) / 8))
    frames = (8 * k) + 1
    actual_seconds = frames / fps
    return frames, k, actual_seconds

def get_ltx_frames_from_k(k: int, fps: float = 24.0) -> tuple[int, int, float]:
    """Calculate frames directly from the integer multiplier k."""
    k = max(1, int(k))
    frames = (8 * k) + 1
    actual_seconds = frames / fps
    return frames, k, actual_seconds

def print_result(frames: int, k: int, actual_seconds: float, fps: float):
    print("\n" + "=" * 42)
    print("        LTX-2.5 FRAME CALCULATION")
    print("=" * 42)
    print(f"  Valid Frame Count : {frames}")
    print(f"  Multiplier (k)    : {k}  (formula: 8 * {k} + 1)")
    print(f"  Actual Duration   : {actual_seconds:.3f} seconds (@ {fps:.1f} fps)")
    print("=" * 42 + "\n")

def print_table(fps: float = 24.0):
    print("\n" + "=" * 55)
    print(f"   LTX-2.5 STANDARD FRAME REFERENCE TABLE (@ {fps:.1f} FPS)")
    print("=" * 55)
    print(f" {'k':^6} | {'num_frames (8k+1)':^20} | {'Duration (s)':^18} ")
    print("-" * 55)
    for k in range(1, 26):
        frames = (8 * k) + 1
        duration = frames / fps
        print(f" {k:^6} | {frames:^20} | {duration:^18.3f} ")
    print("=" * 55 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Calculate valid LTX-2.5 frame counts (satisfies 8k + 1 formula)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--seconds", type=float, help="Target duration in seconds")
    group.add_argument("-f", "--frames", type=int, help="Target rough frame count to snap to nearest 8k+1")
    group.add_argument("-k", "--k-val", type=int, help="Specify integer multiplier k directly")
    group.add_argument("-t", "--table", action="store_true", help="Print reference lookup table (k=1 to 25)")
    
    parser.add_argument("--fps", type=float, default=24.0, help="Frame rate (default: 24.0)")
    args = parser.parse_args()

    # Table flag
    if args.table:
        print_table(args.fps)
        return

    # CLI Flag Mode
    if args.seconds is not None:
        frames, k, actual_sec = get_ltx_frames_from_seconds(args.seconds, args.fps)
        print_result(frames, k, actual_sec, args.fps)
        return
    elif args.frames is not None:
        frames, k, actual_sec = get_ltx_frames_from_raw_frames(args.frames, args.fps)
        print_result(frames, k, actual_sec, args.fps)
        return
    elif args.k_val is not None:
        frames, k, actual_sec = get_ltx_frames_from_k(args.k_val, args.fps)
        print_result(frames, k, actual_sec, args.fps)
        return

    # Interactive Mode (when run with no arguments)
    try:
        user_input = input("Enter target duration in seconds (or add 'f' for frames, e.g., '4s' or '100f'): ").strip()
        if not user_input:
            print("No input provided. Exiting.")
            sys.exit(0)

        if user_input.lower().endswith("f"):
            val = int(user_input[:-1].strip())
            frames, k, actual_sec = get_ltx_frames_from_raw_frames(val, args.fps)
        else:
            val = float(user_input.rstrip("s").strip())
            frames, k, actual_sec = get_ltx_frames_from_seconds(val, args.fps)

        print_result(frames, k, actual_sec, args.fps)
    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as e:
        print(f"Error parsing input: {e}")

if __name__ == "__main__":
    main()
