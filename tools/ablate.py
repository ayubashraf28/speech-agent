# tools/ablate.py
"""
Evaluate speaker-ID stability over a folder of utterances.

Outputs a console table and a CSV with:
- files, unique labels, unknown rate, mean/median similarity

Usage:
  python tools/ablate.py --audio-dir data/live --speaker-db data/speakers/default.npz --thresholds 0.70,0.78,0.82,0.88
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import csv
import numpy as np

# We rely on your wrapper
from utils.speaker_id import SpeakerID

def list_wavs(audio_dir: Path, limit: Optional[int]) -> List[Path]:
    files = sorted(audio_dir.glob("utter_*.wav"))
    return files[:limit] if (limit and limit > 0) else files

def evaluate_once(
    wavs: List[Path],
    db_path: Path,
    threshold: Optional[float],
    margin: float,
) -> Dict[str, object]:
    """
    If threshold is None => baseline "NONE" (speaker-id off): all unknown.
    Otherwise run SpeakerID with given threshold.
    """
    labels = []
    sims: List[float] = []
    unknown = 0

    if threshold is None:
        # Baseline: no speaker ID at all
        for _ in wavs:
            labels.append("Unknown")
            unknown += 1
        return {
            "thr": "NONE",
            "files": len(wavs),
            "unique_labels": len(set(labels)),
            "unknown_rate": round(unknown / max(1, len(wavs)), 3),
            "mean_sim": "",
            "median_sim": "",
        }

    spk = SpeakerID(sample_rate=16000, threshold=float(threshold), margin=float(margin))
    try:
        spk.attach_db(db_path)
        spk.load(db_path)  # ok if empty / new
    except Exception:
        pass

    for w in wavs:
        label, sim = spk.identify(w)
        labels.append(label or "Unknown")
        if not label or label.lower().startswith("unknown"):
            unknown += 1
        else:
            if sim is not None:
                sims.append(sim)

    uniq = len(set(labels))
    mean_sim = round(float(np.mean(sims)), 3) if sims else ""
    median_sim = round(float(np.median(sims)), 3) if sims else ""

    return {
        "thr": float(threshold),
        "files": len(wavs),
        "unique_labels": uniq,
        "unknown_rate": round(unknown / max(1, len(wavs)), 3),
        "mean_sim": mean_sim,
        "median_sim": median_sim,
    }

def main():
    ap = argparse.ArgumentParser(description="Ablate speaker-ID thresholds on a folder of utterances.")
    ap.add_argument("--audio-dir", type=Path, required=True, help="Folder containing utter_*.wav")
    ap.add_argument("--speaker-db", type=Path, required=True, help="NPZ DB used by realtime runs")
    ap.add_argument("--thresholds", type=str, default="0.70,0.78,0.82,0.88", help="Comma-separated thresholds")
    ap.add_argument("--margin", type=float, default=0.08, help="Margin (gap vs 2nd-best)")
    ap.add_argument("--limit", type=int, default=0, help="Use only first N files (0 = all)")
    ap.add_argument("--out-csv", type=Path, default=Path("data/logs/ablation_results.csv"))
    args = ap.parse_args()

    wavs = list_wavs(args.audio_dir, args.limit)
    if not wavs:
        raise SystemExit(f"No wavs found under {args.audio_dir}")

    thresholds: List[Optional[float]] = [None]  # baseline "NONE"
    try:
        thresholds += [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    except Exception:
        raise SystemExit("Bad --thresholds. Example: 0.70,0.78,0.82,0.88")

    rows = []
    print("\nAblation over", len(wavs), "files in", args.audio_dir)
    print("DB:", args.speaker_db)
    print("\nTHR     files  uniq  unknown  mean_sim  median_sim")
    print("-----   -----  ----  -------  --------  ----------")

    for thr in thresholds:
        res = evaluate_once(wavs, args.speaker_db, thr, args.margin)
        rows.append(res)
        print(f"{str(res['thr']).ljust(6)}  {str(res['files']).rjust(5)}  "
              f"{str(res['unique_labels']).rjust(4)}  {str(res['unknown_rate']).rjust(7)}  "
              f"{str(res['mean_sim']).rjust(8)}  {str(res['median_sim']).rjust(10)}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["thr","files","unique_labels","unknown_rate","mean_sim","median_sim"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[ablate] Wrote {args.out_csv}")

if __name__ == "__main__":
    main()
