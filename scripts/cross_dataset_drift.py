"""Cross-dataset calibration drift experiment (E8).

Fit calibrators + fusion on subset A's calibration split, evaluate on
subset B's test split. Reports the calibration-drift diagnostic
statistic and the actual calibration loss to validate the diagnostic's
predictive power.

This is the empirical support for the drift diagnostic and the §Discussion
limitation about calibrators not transferring across distributions.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.diagnostics.calibration_drift import calibration_drift_test  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from scripts.eval_from_npz import build_methods  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def load(path: Path):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    graded = data["graded"].astype(np.float64)
    qids = list(data["qids"])
    split = np.asarray(data["split"])
    return X, y, graded, qids, split


def run(src_path, dst_path, methods_set):
    Xs, ys, gs, qids_s, split_s = load(src_path)
    Xt, yt, gt, qids_t, split_t = load(dst_path)
    cal_s = split_s == "calibration"
    test_t = split_t == "test"

    qids_cal = [qids_s[i] for i in range(len(qids_s)) if cal_s[i]]
    qids_test = [qids_t[i] for i in range(len(qids_t)) if test_t[i]]

    drift = calibration_drift_test(Xs[cal_s], Xt[test_t], alpha=0.05)
    drift_stats = {SIGNAL_ORDER[j]: float(drift.per_signal_statistics[j])
                   for j in range(len(SIGNAL_ORDER))}

    signal_cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    methods = build_methods(signal_cols)
    out = {"drift_per_signal_ks": drift_stats,
           "drift_reject": bool(drift.reject),
           "worst_drift_signal": SIGNAL_ORDER[int(drift.worst_signal)],
           "methods": {}}
    for name, fusion in methods.items():
        if methods_set and name not in methods_set:
            continue
        try:
            fusion.fit(Xs[cal_s], ys[cal_s], query_ids=qids_cal)
            p = fusion.fuse(Xt[test_t], query_ids=qids_test)
            ev = evaluate(p, yt[test_t], graded_labels=gt[test_t],
                          query_ids=qids_test, include_reliability=False).as_dict()
            out["methods"][name] = ev
        except Exception as e:
            out["methods"][name] = {"error": repr(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="eval")
    ap.add_argument("--subsets", nargs="+", default=["nfcorpus", "scifact", "arguana"])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--out", default="eval/cross_dataset_drift.json")
    args = ap.parse_args()

    method_set = set(args.methods) if args.methods else None
    npz_paths = {s: Path(args.npz_dir) / f"beir_{s}_results.npz" for s in args.subsets}
    for s, p in npz_paths.items():
        if not p.exists():
            sys.exit(f"missing npz for {s}: {p}")

    aggregate = {}
    for src, dst in permutations(args.subsets, 2):
        key = f"{src}->{dst}"
        print(f"\n=== {key} ===")
        res = run(npz_paths[src], npz_paths[dst], method_set)
        aggregate[key] = res
        print(f"  drift KS reject={res['drift_reject']}  worst_signal={res['worst_drift_signal']}")
        for m, ev in res["methods"].items():
            if "error" in ev:
                print(f"  {m:<24} ERROR")
                continue
            print(f"  {m:<24} ECE15={ev['ece_15']:.4f}  NDCG@10={ev['ndcg_10'] or 0:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
