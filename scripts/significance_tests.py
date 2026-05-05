"""Paired bootstrap significance tests on per-query NDCG@10 deltas.

For each (subset, baseline) pair we resample queries with replacement,
compute the per-query NDCG@10 of CalFuse-Conformal vs the baseline on
each resample, and report the mean delta plus 95% percentile CI.

The unit of resampling is the query, not the (query, passage) pair --
NDCG@10 is a query-level metric and pair-level resampling would
artificially shrink the variance.
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

from src.evaluate import ndcg_at_k  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402
from src.fusion.calfuse_conformal import ConformalCalFuse  # noqa: E402
from src.fusion.linear_learned import LinearLearnedFusion  # noqa: E402
from src.fusion.multicalibration import (  # noqa: E402
    Multicalibration,
    signal_dominance_subgroups,
)
from src.fusion.single_signal import SingleSignalFusion  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def per_query_ndcg(probs, graded, qids, k=10):
    by_q = {}
    for i, q in enumerate(qids):
        by_q.setdefault(q, []).append(i)
    out = {}
    for q, idxs in by_q.items():
        idxs = np.array(idxs, dtype=np.int64)
        s = probs[idxs]
        y = graded[idxs]
        order = np.argsort(-s, kind="mergesort")[:k]
        y_ranked = y[order]
        discounts = 1.0 / np.log2(np.arange(2, len(y_ranked) + 2))
        dcg = float(np.sum((2 ** y_ranked - 1) * discounts))
        ideal_order = np.argsort(-y, kind="mergesort")[:k]
        y_ideal = y[ideal_order]
        discounts_i = 1.0 / np.log2(np.arange(2, len(y_ideal) + 2))
        idcg = float(np.sum((2 ** y_ideal - 1) * discounts_i))
        if idcg > 0:
            out[q] = dcg / idcg
    return out


def make_methods():
    bm25_col = SIGNAL_ORDER.index("bm25")
    bge_col = SIGNAL_ORDER.index("dense_bge")
    e5_col = SIGNAL_ORDER.index("dense_e5")
    return {
        "bm25_platt": SingleSignalFusion(bm25_col, name="bm25_platt"),
        "bge_platt": SingleSignalFusion(bge_col, name="bge_platt"),
        "e5_platt": SingleSignalFusion(e5_col, name="e5_platt"),
        "linear_learned": LinearLearnedFusion(),
        "calfuse_conformal": ConformalCalFuse(
            base=CalFuseFusion(force_mode="parametric"),
            subgroup_fn=signal_dominance_subgroups(),
        ),
    }


def run(npz_path: Path, n_boot=2000, seed=2026):
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

    methods = make_methods()
    pq = {}
    for name, fusion in methods.items():
        fusion.fit(X[cal], y[cal], query_ids=qids_cal)
        p = fusion.fuse(X[test], query_ids=qids_test)
        pq[name] = per_query_ndcg(p, graded[test], qids_test, k=10)

    test_qs = sorted(set(qids_test))
    rng = np.random.default_rng(seed)
    out = {"n_test_queries": len(test_qs), "vs_conformal": {}}
    target = "calfuse_conformal"
    for baseline in [m for m in methods if m != target]:
        deltas = []
        for q in test_qs:
            if q in pq[target] and q in pq[baseline]:
                deltas.append(pq[target][q] - pq[baseline][q])
        deltas = np.array(deltas, dtype=np.float64)
        # Paired bootstrap.
        boots = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, len(deltas), size=len(deltas))
            boots[b] = deltas[idx].mean()
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
        # Two-sided p-value for H0: mean delta = 0 (sign-test variant).
        p_val = float(2.0 * min((boots <= 0).mean(), (boots >= 0).mean()))
        out["vs_conformal"][baseline] = {
            "n_paired_queries": len(deltas),
            "mean_delta_ndcg": float(deltas.mean()),
            "ci95_lo": float(ci_lo),
            "ci95_hi": float(ci_hi),
            "boot_p_value": p_val,
            "frac_queries_conformal_better": float((deltas > 0).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="eval/significance_tests.json")
    args = ap.parse_args()

    aggregate = {}
    for path in args.npz:
        name = Path(path).stem.replace("beir_", "").replace("_results", "")
        print(f"\n=== {name} ===")
        res = run(Path(path), n_boot=args.n_boot)
        aggregate[name] = res
        print(f"  n_queries={res['n_test_queries']}")
        for b, v in res["vs_conformal"].items():
            sig = "**" if (v["ci95_lo"] > 0 or v["ci95_hi"] < 0) else "  "
            print(f"  {sig} conformal vs {b:<22}  Δ={v['mean_delta_ndcg']:+.4f}  "
                  f"95% CI=[{v['ci95_lo']:+.4f}, {v['ci95_hi']:+.4f}]  "
                  f"p={v['boot_p_value']:.4f}  win={v['frac_queries_conformal_better']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
