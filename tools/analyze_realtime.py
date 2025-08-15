# tools/analyze_realtime.py
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

LIVE = Path("data/live")
INDEX = LIVE / "realtime_index.csv"
REPORT = LIVE / "analysis_summary.md"

def load_rows():
    if not INDEX.exists():
        raise FileNotFoundError(f"Missing {INDEX}. Run realtime_agent first.")
    rows = []
    with INDEX.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["sim"] = float(row.get("sim", "") or 0.0)
            rows.append(row)
    return rows

def main():
    rows = load_rows()
    n = len(rows)
    labels = [r["label"] or "" for r in rows]
    texts = [r["text"] for r in rows]
    sims = [r["sim"] for r in rows if r.get("sim", 0.0) > 0]

    # Basic counts
    label_counts = Counter(labels)
    unique_labels = [k for k in label_counts if k]
    num_unknown = label_counts.get("", 0) + label_counts.get("Unknown", 0)

    # Transitions (how often the assigned label changes turn-to-turn)
    transitions = 0
    for i in range(1, n):
        if labels[i] != labels[i-1]:
            transitions += 1

    # Average run length per label (how “sticky” is a label once assigned)
    runs = defaultdict(list)
    cur_label, run_len = None, 0
    for lab in labels:
        if lab == cur_label:
            run_len += 1
        else:
            if cur_label is not None:
                runs[cur_label].append(run_len)
            cur_label, run_len = lab, 1
    if cur_label is not None:
        runs[cur_label].append(run_len)

    avg_run = {lab: round(mean(lens), 2) for lab, lens in runs.items()}
    avg_sim = round(mean(sims), 3) if sims else 0.0

    # Name-claim heuristics: how often people said “my name is … / call me … / this is …”
    def claimed_name(t: str) -> bool:
        t = (t or "").lower()
        return ("my name is " in t) or ("call me " in t) or ("this is " in t)

    claim_rate = sum(1 for t in texts if claimed_name(t))
    claim_rate_pct = round(100 * claim_rate / n, 1) if n else 0.0

    # Build report
    lines = []
    lines.append("# Real-time Agent: Session Analysis\n")
    lines.append(f"- Utterances: **{n}**")
    lines.append(f"- Unique labels: **{len(unique_labels)}** → {sorted(unique_labels)}")
    lines.append(f"- Unknown/empty labels: **{num_unknown}**")
    lines.append(f"- Avg similarity (where available): **{avg_sim}**")
    lines.append(f"- Label transitions: **{transitions}** (out of {max(0, n-1)} steps)")
    lines.append(f"- Avg run length per label: `{avg_run}`")
    lines.append(f"- Utterances with explicit name claim: **{claim_rate}** ({claim_rate_pct}%)\n")

    lines.append("## Label counts")
    for lab, c in label_counts.most_common():
        show = lab if lab else "(empty)"
        lines.append(f"- {show}: {c}")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[analysis] Wrote {REPORT}")

if __name__ == "__main__":
    main()
