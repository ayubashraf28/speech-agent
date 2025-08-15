# scripts/evaluate_corpus.py
import argparse, csv, difflib, statistics, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

def fuzzy_eq(a,b,thr=0.8):
    a=(a or "").strip().lower(); b=(b or "").strip().lower()
    if not a or not b: return False
    return difflib.SequenceMatcher(None,a,b).ratio()>=thr

def load_manifest(path: Path):
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        rd=csv.DictReader(f)
        for r in rd:
            r["pred_sim"]=float(r.get("pred_sim","0") or 0)
            rows.append(r)
    return rows

def eval_one(path: Path):
    rows=load_manifest(path)
    n=len(rows)
    sims=[r["pred_sim"] for r in rows if r["pred_sim"]>=0]
    unknown=sum(1 for r in rows if (r.get("pred_label","") or "").lower().startswith("guest"))
    claimed=[r for r in rows if r.get("claimed_name","").strip()]
    # stability on claimed
    stable=sum(1 for r in claimed if fuzzy_eq(r["claimed_name"], r.get("pred_label","")))
    by_claim=defaultdict(lambda: {"tot":0,"ok":0})
    for r in claimed:
        c=r["claimed_name"]; by_claim[c]["tot"]+=1
        if fuzzy_eq(c, r.get("pred_label","")): by_claim[c]["ok"]+=1
    # confusion (only on claimed)
    names=sorted(set(r["claimed_name"] for r in claimed))
    labelset=sorted(set(r.get("pred_label","") for r in claimed))
    mat=np.zeros((len(names), len(labelset)), dtype=int)
    name2i={n:i for i,n in enumerate(names)}
    lab2j={l:j for j,l in enumerate(labelset)}
    for r in claimed:
        i=name2i[r["claimed_name"]]; j=lab2j[r.get("pred_label","")]
        mat[i,j]+=1

    stats={
        "file": str(path),
        "utterances": n,
        "speakers_pred": len(set(r.get("pred_label","") for r in rows)),
        "avg_sim": round(float(np.mean(sims)),4) if sims else 0.0,
        "med_sim": round(float(np.median(sims)),4) if sims else 0.0,
        "unknown_rate": round(unknown/max(1,n),4),
        "claimed_utts": len(claimed),
        "claimed_stability": round(stable/max(1,len(claimed)),4),
        "per_claim": {k: round(v["ok"]/max(1,v["tot"]),4) for k,v in by_claim.items()},
        "confusion": (names, labelset, mat),
    }
    return stats

def print_summary(stats):
    print(f"\n=== {stats['file']} ===")
    print(f"Utterances: {stats['utterances']}")
    print(f"Pred speakers: {stats['speakers_pred']}")
    print(f"Avg/Med similarity: {stats['avg_sim']} / {stats['med_sim']}")
    print(f"Unknown rate: {stats['unknown_rate']}")
    print(f"Claimed utts: {stats['claimed_utts']}")
    print(f"Stability on claimed: {stats['claimed_stability']}")
    if stats["per_claim"]:
        print("Per-claimed accuracy:")
        for k,v in stats["per_claim"].items():
            print(f"  - {k}: {v}")

def save_confusion(stats, outdir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    names, labels, M = stats["confusion"]
    if not len(names) or not len(labels): return
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("Claimed name")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p = outdir / (Path(stats["file"]).parent.name + "_confusion.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"Saved confusion: {p}")

def main():
    ap=argparse.ArgumentParser("Evaluate one or more manifests")
    ap.add_argument("manifests", nargs="+", type=Path, help="paths to manifest.csv")
    ap.add_argument("--out", type=Path, default=Path("corpus/eval_out"))
    args=ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    allstats=[]
    for m in args.manifests:
        s=eval_one(m)
        print_summary(s)
        save_confusion(s, args.out)
        allstats.append(s)
    if len(allstats)>=2:
        # quick side-by-side table for ablation
        print("\n=== Comparison (first vs others) ===")
        base=allstats[0]
        keys=["utterances","speakers_pred","avg_sim","med_sim","unknown_rate","claimed_utts","claimed_stability"]
        header=["metric", Path(base["file"]).parent.name] + [Path(s["file"]).parent.name for s in allstats[1:]]
        print("\t".join(header))
        for k in keys:
            row=[k, str(base[k])] + [str(s[k]) for s in allstats[1:]]
            print("\t".join(row))

if __name__=="__main__":
    main()
