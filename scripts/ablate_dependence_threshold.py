"""Sensitivity sweep on the CalFuse auto-mode dependence threshold.

Sweeps the threshold over {0.1, 0.2, 0.3, 0.5} and reports the
parametric-vs-learned mode selection that results, plus ECE / NDCG of
the chosen mode on each BEIR test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluate import evaluate  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def run_one(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    graded = data["graded"].astype(np.float64)
    qids = list(data["qids"])
    split = np.asarray(data["split"])
    cal = split == "calibration"
    test = split == "test"
    qids_cal = [qids[i] for i in range(len(qids)) if cal[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test[i]]

    out = {}
    for thr in [0.1, 0.2, 0.3, 0.5]:
        # `force_mode=None` lets the threshold drive the choice.
        f = CalFuseFusion(dependence_threshold=thr, force_mode=None)
        f.fit(X[cal], y[cal], query_ids=qids_cal)
        p = f.fuse(X[test], query_ids=qids_test)
        ev = evaluate(p, y[test], graded_labels=graded[test], query_ids=qids_test,
                      include_reliability=False).as_dict()
        out[f"{thr:.2f}"] = {
            "mode": f.report_.mode,
            "dep_score": float(f.report_.dependence_score),
            "ece_15": ev["ece_15"],
            "ndcg_10": ev["ndcg_10"],
            "brier": ev["brier"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+",
                    default=[f"eval/beir_{s}_results.npz" for s in
                             ["nfcorpus", "scifact", "fiqa", "trec-covid", "arguana"]])
    ap.add_argument("--out", default="eval/ablate_dependence_threshold.json")
    args = ap.parse_args()

    aggregate = {}
    for path in args.npz:
        name = Path(path).stem.replace("beir_", "").replace("_results", "")
        if not Path(path).exists():
            print(f"skip {name}: {path}")
            continue
        print(f"\n=== {name} ===")
        res = run_one(Path(path))
        aggregate[name] = res
        for thr, r in res.items():
            print(f"  thr={thr}  mode={r['mode']:<10} dep={r['dep_score']:.3f}  "
                  f"ECE={r['ece_15']:.4f}  NDCG@10={r['ndcg_10']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
