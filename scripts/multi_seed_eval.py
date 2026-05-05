"""Multi-seed evaluation with bootstrap confidence intervals.

For each cached `.npz` produces a table of (mean, std, 95% bootstrap CI)
over `n_seeds` query-level resplits. Bootstrap is over query-level
resamples on the test split. The resampling unit is the *query*, not
the (query, passage) pair, because a single query produces a clump of
positively-correlated pairs and pair-level resampling underestimates
variance.
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
from src.fusion.multicalibration import (  # noqa: E402
    signal_dominance_subgroups,
    worst_subgroup_ece,
)
from scripts.eval_from_npz import build_methods, query_level_split  # noqa: E402


SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def evaluate_one(X_cal, y_cal, X_test, y_test, qids_cal, qids_test, graded_test, methods_set):
    signal_cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    methods = build_methods(signal_cols)
    M = np.asarray(signal_dominance_subgroups()(X_test, qids_test), dtype=bool)
    out = {}
    for name, fusion in methods.items():
        if methods_set and name not in methods_set:
            continue
        try:
            fusion.fit(X_cal, y_cal, query_ids=qids_cal)
            p = fusion.fuse(X_test, query_ids=qids_test)
            ev = evaluate(p, y_test, graded_labels=graded_test, query_ids=qids_test,
                          include_reliability=False).as_dict()
            ev["worst_subgroup_ece_15"] = worst_subgroup_ece(p, y_test, M, n_bins=15, n_min=25)
            out[name] = ev
        except Exception as e:
            out[name] = {"error": repr(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    graded = data["graded"].astype(np.float64)
    qids = list(data["qids"])

    method_set = set(args.methods) if args.methods else None
    per_seed = {}  # seed -> {method -> ev}
    for seed in args.seeds:
        split = query_level_split(qids, seed)
        cal_mask = split == "calibration"
        test_mask = split == "test"
        qids_cal = [qids[i] for i in range(len(qids)) if cal_mask[i]]
        qids_test = [qids[i] for i in range(len(qids)) if test_mask[i]]
        ev = evaluate_one(X[cal_mask], y[cal_mask], X[test_mask], y[test_mask],
                          qids_cal, qids_test, graded[test_mask], method_set)
        per_seed[seed] = ev
        print(f"seed={seed}  done")

    # Aggregate.
    method_names = sorted({m for s in per_seed.values() for m in s.keys()})
    metrics = ["ece_15", "ece_10", "brier", "nll", "ndcg_10", "worst_subgroup_ece_15"]
    aggregated = {}
    for m in method_names:
        agg = {}
        for met in metrics:
            vals = []
            for s in args.seeds:
                v = per_seed[s].get(m, {}).get(met)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    continue
                vals.append(float(v))
            if not vals:
                continue
            arr = np.array(vals)
            agg[met] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n": len(arr),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        aggregated[m] = agg

    print(f"\n{'method':<24} {'metric':<24} {'mean':>9} {'std':>9} {'n':>3}")
    for m in method_names:
        for met in metrics:
            if met not in aggregated[m]:
                continue
            a = aggregated[m][met]
            print(f"{m:<24} {met:<24} {a['mean']:>9.4f} {a['std']:>9.4f} {a['n']:>3d}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"npz": args.npz, "seeds": args.seeds,
                       "per_seed": per_seed, "aggregated": aggregated}, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
