# tools/eval_realtime.py
import csv
import math
from pathlib import Path
from statistics import mean

CSV_PATH = Path("data/live/realtime_index.csv")

def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d

def quantiles(xs, qs=(0.5, 0.9, 0.95)):
    if not xs:
        return {q: float("nan") for q in qs}
    xs = sorted(xs)
    out = {}
    for q in qs:
        k = (len(xs) - 1) * q
        f = math.floor(k); c = math.ceil(k)
        if f == c:
            out[q] = xs[int(k)]
        else:
            out[q] = xs[f] + (xs[c] - xs[f]) * (k - f)
    return out

def load_rows(path: Path):
    rows = []
    if not path.exists():
        print(f"No CSV found: {path}")
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def is_unknown(label: str) -> bool:
    if not label:
        return True
    L = label.lower()
    return L.startswith("guest") or L == "unknown"

def claimed_name(text: str) -> str:
    import re
    m = re.search(r"\b(i am|i'm|this is|my name is|call me)\s+([A-Za-z][A-Za-z\-']{1,30})\b", (text or ""), flags=re.I)
    return m.group(2).strip().title() if m else ""

def main():
    rows = load_rows(CSV_PATH)
    if not rows:
        return

    N = len(rows)
    sims = [float(r.get("sim", "0") or 0) for r in rows]
    asr = [int(r.get("asr_ms", "0") or 0) for r in rows]
    sid = [int(r.get("id_ms", "0") or 0) for r in rows]
    llm = [int(r.get("llm_ms", "0") or 0) for r in rows]
    tts = [int(r.get("tts_ms", "0") or 0) for r in rows]
    total = [int(r.get("total_pipeline_ms", "0") or 0) for r in rows]
    audio_ms = [int(r.get("audio_len_ms", "0") or 0) for r in rows]

    labels = [r.get("label", "") for r in rows]
    unknowns = sum(1 for l in labels if is_unknown(l))
    unique_labels = sorted(set(labels))
    guest_labels = sorted({l for l in unique_labels if l.lower().startswith("guest")})

    # Name-claim accuracy: if the text contains a claimed name, did label match?
    claims = []
    for r in rows:
        c = r.get("claimed_name") or claimed_name(r.get("text", ""))
        if c:
            claims.append((c, r.get("label", "")))
    claim_total = len(claims)
    claim_correct = sum(1 for c, lab in claims if c.lower() == (lab or "").lower())

    # Fragmentation proxy: how many distinct Guest-* labels vs. number of turns
    guest_turns = sum(1 for l in labels if l.lower().startswith("guest"))
    frag = len(guest_labels)

    print("=== Quantitative Report (realtime_index.csv) ===")
    print(f"Turns: {N}")
    print(f"Audio len (ms): mean={int(mean(audio_ms) if audio_ms else 0)} "
          f"p50={int(quantiles(audio_ms)[0.5]) if audio_ms else 0} "
          f"p90={int(quantiles(audio_ms)[0.9]) if audio_ms else 0}")
    print()
    print(f"[Labeling] unknown rate: {pct(unknowns, N):.1f}%  "
          f"(unknown={unknowns}/{N}), unique labels={len(unique_labels)}, guest clusters={frag}")
    if claim_total:
        print(f"[Labeling] name-claim accuracy: {pct(claim_correct, claim_total):.1f}%  "
              f"({claim_correct}/{claim_total})")
    else:
        print("[Labeling] name-claim accuracy: n/a (no explicit name claims)")
    if sims:
        q = quantiles(sims)
        print(f"[Labeling] similarity: mean={mean(sims):.3f}  p50={q[0.5]:.3f}  p90={q[0.9]:.3f}  p95={q[0.95]:.3f}")
    print()
    def line(title, xs):
        if not xs:
            print(f"[Timing] {title}: n/a"); return
        q = quantiles(xs)
        print(f"[Timing] {title}: mean={int(mean(xs))} ms  p50={int(q[0.5])}  p90={int(q[0.9])}  p95={int(q[0.95])}")
    line("ASR", asr)
    line("Speaker-ID", sid)
    line("LLM", llm)
    line("TTS", tts)
    line("TOTAL pipeline", total)
    print("===============================================")

if __name__ == "__main__":
    main()
