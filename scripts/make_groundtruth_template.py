# scripts/make_groundtruth_template.py
"""
Scan data/live/, collect utter_*.wav files, run current SpeakerID offline to
produce a 'suggested_label' + similarity, and write a template CSV you can
quickly fill with ground-truth labels.

Output:
  data/eval/groundtruth.csv with columns:
    wav, timestamp, suggested_label, sim, true_label, notes

Usage:
  python scripts/make_groundtruth_template.py \
    --speaker-db data/speakers/default.npz
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.speaker_id import SpeakerID  # type: ignore

LIVE_DIR = Path("data/live")
EVAL_DIR = Path("data/eval")


def safe_identify(spk: SpeakerID, wav: Path) -> Tuple[str, float]:
    try:
        out = spk.identify(wav)
        if isinstance(out, tuple):
            label = str(out[0])
            sim = float(out[1]) if len(out) > 1 else 0.0
        else:
            label = str(out)
            sim = 0.0
        return label, sim
    except Exception:
        return "Unknown", 0.0


def main(speaker_db: Optional[Path]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    spk = SpeakerID()  # uses your defaults
    if speaker_db and speaker_db.exists():
        try:
            spk.load(speaker_db)  # type: ignore[attr-defined]
            print(f"[Eval] Loaded speaker DB: {speaker_db}")
        except Exception as e:
            print(f"[Eval] Could not load DB ({speaker_db}): {e}")

    wavs = sorted(LIVE_DIR.glob("utter_*.wav"))
    if not wavs:
        print("[Eval] No WAV files found in data/live/.")
        return

    out_csv = EVAL_DIR / "groundtruth.csv"
    rows = []
    for wav in wavs:
        label, sim = safe_identify(spk, wav)
        ts = wav.stem.replace("utter_", "")
        rows.append({
            "wav": str(wav),
            "timestamp": ts,
            "suggested_label": label,
            "sim": f"{sim:.2f}",
            "true_label": "",
            "notes": "",
        })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["wav", "timestamp", "suggested_label", "sim", "true_label", "notes"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Eval] Wrote {out_csv} ({len(rows)} rows). Fill 'true_label' where you can.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speaker-db", type=Path, default=Path("data/speakers/default.npz"))
    args = p.parse_args()
    main(args.speaker_db)
