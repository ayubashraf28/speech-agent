# diarize_test.py
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from asr.diarize import (
    diarize_wav,
    save_segments_csv,
    save_rttm,
    merge_contiguous,
)

# Helpful defaults for Windows and shared drives
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("SPEECHBRAIN_STRATEGY", "copy")

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Quick diarization test harness")
    p.add_argument(
        "--file",
        type=Path,
        default=Path("data/test/demo.wav"),
        help="Path to test audio file (wav/mp3).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/test"),
        help="Where to save CSV/RTTM outputs.",
    )
    p.add_argument(
        "--merge-gap",
        type=float,
        default=0.2,
        help="Merge contiguous same-speaker segments with gaps ≤ this many seconds.",
    )
    p.add_argument(
        "--min-dur",
        type=float,
        default=0.25,
        help="Drop merged segments shorter than this duration (seconds).",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge contiguous segments; save raw diarization.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.file.exists():
        print(f"[Diarize Test] File not found: {args.file}")
        sys.exit(1)

    print(f"[Diarize Test] Using: {args.file}")

    try:
        segments = diarize_wav(str(args.file))
    except Exception as e:
        print(f"[Diarize Test] Diarization failed: {e}")
        print("Tip: ensure HUGGINGFACE_TOKEN (or HF_TOKEN) is set in your environment.")
        sys.exit(2)

    print(f"[Diarize Test] Raw segments: {len(segments)}")

    if args.no_merge:
        merged = segments
    else:
        merged = merge_contiguous(
            segments,
            gap_threshold=args.merge_gap,
            min_duration=args.min_dur,
        )

    print(f"[Diarize Test] Merged segments: {len(merged)}")

    # Prepare outputs
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.file.stem
    csv_path = args.outdir / f"{stem}_diarization.csv"
    rttm_path = args.outdir / f"{stem}.rttm"

    # Save files
    save_segments_csv(merged, str(csv_path))
    save_rttm(merged, str(rttm_path), uri=stem)

    # Print first few lines for sanity
    for s in merged[:10]:
        print(f"{s.start:.2f}–{s.end:.2f} {s.speaker}")

    print(f"[Diarize Test] Saved CSV:  {csv_path}")
    print(f"[Diarize Test] Saved RTTM: {rttm_path}")


if __name__ == "__main__":
    main()
