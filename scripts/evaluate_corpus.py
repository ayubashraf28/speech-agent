# scripts/evaluate_corpus.py
"""
Read data/eval/groundtruth.csv (you can partially fill true_label),
compute speaker-ID metrics and approximate latency, and emit:

  data/eval/metrics.json
  data/eval/confusion_matrix.png
  data/eval/latency_hist.png
  data/eval/report.md
  data/eval/predictions.csv  (per-utterance summary)

Assumptions:
- Utterance WAVs live in data/live/ as utter_*.wav
- Reply artifacts are either utter_..._reply.mp3 or utter_..._reply.txt
  (the code uses file modification times to approximate round-trip latency)

Usage:
  python scripts/evaluate_corpus.py --speaker-db data/speakers/default.npz
"""

from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Project imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.speaker_id import SpeakerID  # type: ignore


LIVE_DIR = Path("data/live")
EVAL_DIR = Path("data/eval")


def approx_latency_ms(wav: Path) -> Optional[float]:
    """
    Approximate total round-trip latency by comparing last mod-time
    of the reply artifact (mp3 or txt) with the wav mod-time.
    """
    if not wav.exists():
        return None
    wav_mtime = wav.stat().st_mtime
    mp3 = wav.with_name(wav.stem + "_reply.mp3")
    txt = wav.with_name(wav.stem + "_reply.txt")
    reply = mp3 if mp3.exists() else txt if txt.exists() else None
    if reply is None:
        return None
    rp_mtime = reply.stat().st_mtime
    return max(0.0, (rp_mtime - wav_mtime) * 1000.0)


def load_groundtruth(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run make_groundtruth_template.py first.")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def evaluate(rows: List[Dict[str, str]]) -> Tuple[Dict, List[Dict]]:
    """
    Compute:
      - accuracy (where true_label is provided)
      - unknown rate (pred label startswith 'Guest' or 'Unknown')
      - confusion counts
      - label stability per true speaker (num predicted labels per true)
      - latency stats (approx via mtime diff)
    """
    preds = []
    conf = defaultdict(Counter)
    known = correct = 0
    unknown_pred = 0
    stability_map = defaultdict(list)
    latencies = []

    for r in rows:
        wav = Path(r["wav"])
        pred = (r.get("suggested_label") or "").strip() or "Unknown"
        true = (r.get("true_label") or "").strip()
        sim = r.get("sim") or ""
        lat = approx_latency_ms(wav)
        if lat is not None:
            latencies.append(lat)

        preds.append({
            "wav": str(wav),
            "predicted_label": pred,
            "true_label": true,
            "sim": sim,
            "latency_ms": f"{lat:.0f}" if lat is not None else "",
        })

        if pred.lower().startswith("guest") or pred.lower().startswith("unknown"):
            unknown_pred += 1

        if true:
            known += 1
            if pred.lower() == true.lower():
                correct += 1
            conf[true][pred] += 1
            stability_map[true].append(pred)

    acc = (correct / known) if known else None
    unk_rate = (unknown_pred / len(rows)) if rows else None
    stability = {
        t: len(set(preds)) for t, preds in stability_map.items()
    }

    # Latency summary
    lat_summary = {}
    if latencies:
        lat_summary = {
            "count": len(latencies),
            "mean_ms": round(statistics.mean(latencies), 1),
            "median_ms": round(statistics.median(latencies), 1),
            "p90_ms": round(statistics.quantiles(latencies, n=10)[8], 1) if len(latencies) >= 10 else None,
            "min_ms": round(min(latencies), 1),
            "max_ms": round(max(latencies), 1),
        }

    metrics = {
        "num_utterances": len(rows),
        "num_with_truth": known,
        "speaker_id_accuracy": round(acc * 100, 2) if acc is not None else None,
        "unknown_prediction_rate": round(unk_rate * 100, 2) if unk_rate is not None else None,
        "stability_labels_per_true": stability,  # e.g., {'Alice': 1, 'Ryan': 1}
        "latency_summary_ms": lat_summary,
        "confusion": {t: dict(c) for t, c in conf.items()},
    }
    return metrics, preds


def plot_confusion(confusion: Dict[str, Dict[str, int]], out: Path) -> None:
    if not confusion:
        return
    trues = sorted(confusion.keys())
    preds = sorted({p for d in confusion.values() for p in d.keys()})

    import numpy as np
    mat = []
    for t in trues:
        row = [confusion[t].get(p, 0) for p in preds]
        mat.append(row)
    mat = np.array(mat, dtype=float)
    # normalize rows
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums

    plt.figure(figsize=(max(6, 0.8 * len(preds)), max(4, 0.6 * len(trues))))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Row-normalized frequency")
    plt.xticks(range(len(preds)), preds, rotation=45, ha="right")
    plt.yticks(range(len(trues)), trues)
    plt.title("Speaker-ID Confusion Matrix (normalized by true)")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def plot_latency_hist(latencies: List[float], out: Path) -> None:
    if not latencies:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(latencies, bins=20)
    plt.xlabel("Approx round-trip latency (ms)")
    plt.ylabel("Count")
    plt.title("Latency distribution (WAV→reply file mtime)")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def write_report(md_path: Path, metrics: Dict, confusion_png: Optional[Path], latency_png: Optional[Path]) -> None:
    md = []
    md.append("# Evaluation Report\n")
    md.append("Generated by `scripts/evaluate_corpus.py`.\n\n")
    md.append("## Summary\n")
    md.append(f"- Utterances: **{metrics['num_utterances']}**\n")
    md.append(f"- With ground-truth labels: **{metrics['num_with_truth']}**\n")
    if metrics["speaker_id_accuracy"] is not None:
        md.append(f"- Speaker-ID accuracy (on labeled subset): **{metrics['speaker_id_accuracy']}%**\n")
    if metrics["unknown_prediction_rate"] is not None:
        md.append(f"- Unknown/Guest prediction rate: **{metrics['unknown_prediction_rate']}%**\n")

    lat = metrics.get("latency_summary_ms") or {}
    if lat:
        md.append("\n## Latency (approximate)\n")
        md.append(f"- Count: {lat.get('count')}\n")
        md.append(f"- Mean: {lat.get('mean_ms')} ms; Median: {lat.get('median_ms')} ms; P90: {lat.get('p90_ms')} ms\n")
        md.append(f"- Min: {lat.get('min_ms')} ms; Max: {lat.get('max_ms')} ms\n")

    stab = metrics.get("stability_labels_per_true") or {}
    if stab:
        md.append("\n## Stability (distinct predicted labels per true speaker)\n")
        for t, k in stab.items():
            md.append(f"- {t}: {k}\n")

    conf = metrics.get("confusion") or {}
    if conf:
        md.append("\n## Confusion Matrix\n")
        if confusion_png and confusion_png.exists():
            md.append(f"![Confusion Matrix]({confusion_png.name})\n")

    if latency_png and latency_png.exists():
        md.append("\n## Latency Histogram\n")
        md.append(f"![Latency Histogram]({latency_png.name})\n")

    md.append("\n## Notes\n")
    md.append("- Latency is approximated from file modified times (WAV vs reply file). Use as a rough indicator, not exact.\n")
    md.append("- Accuracy is computed only on rows where you filled `true_label`.\n")
    md.append("- `Unknown` includes predictions like `Guest-N` or literal `Unknown`.\n")

    md_path.write_text("".join(md), encoding="utf-8")


def main(gt_csv: Path, speaker_db: Optional[Path]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_groundtruth(gt_csv)
    metrics, preds = evaluate(rows)

    # Save per-utterance predictions
    preds_csv = EVAL_DIR / "predictions.csv"
    with preds_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["wav", "predicted_label", "true_label", "sim", "latency_ms"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(preds)

    # Save metrics JSON
    metrics_json = EVAL_DIR / "metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Plots
    confusion_png = EVAL_DIR / "confusion_matrix.png"
    latency_png = EVAL_DIR / "latency_hist.png"

    # Confusion matrix data
    conf = metrics.get("confusion") or {}
    plot_confusion(conf, confusion_png)

    # Latency list comes via preds rows
    latencies = [float(p["latency_ms"]) for p in preds if p["latency_ms"]]
    plot_latency_hist(latencies, latency_png)

    # Markdown report
    report_md = EVAL_DIR / "report.md"
    write_report(report_md, metrics, confusion_png, latency_png)

    print(f"[Eval] Wrote:\n- {preds_csv}\n- {metrics_json}\n- {confusion_png}\n- {latency_png}\n- {report_md}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--groundtruth", type=Path, default=Path("data/eval/groundtruth.csv"))
    p.add_argument("--speaker-db", type=Path, default=Path("data/speakers/default.npz"))
    args = p.parse_args()
    main(args.groundtruth, args.speaker_db)
