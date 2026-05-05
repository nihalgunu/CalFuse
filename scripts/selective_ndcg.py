"""Selective NDCG@10 vs query coverage.

This is the downstream-decisions experiment that turns the
calibration story into a system-level claim. For each fusion method:

1. Compute fused scores on the test split.
2. For each test query, take the max fused score over its candidate
   pool as the *query confidence* (the system's self-reported
   confidence in answering this query at all). For Conformal-CalFuse
   we also offer envelope-width as an alternative confidence signal.
3. Sort queries by confidence (high to low). For each coverage level
   c in [0, 1], the system answers the top c-fraction of queries
   (returning the ranked candidate pool) and abstains on the rest.
4. Report mean NDCG@10 over the answered queries vs c.

A RAG system that uses CalFuse would do exactly this: retrieve and
answer when confident, escalate / refuse when not. The "selective
NDCG-vs-coverage" curve is the right metric for that decision
because it isolates the *quality of selected queries* rather than
the average quality.

We fit per method on the cached signal matrices and produce per-subset
JSON + an aggregate operating-point table. AUC under the
selective-NDCG curve is the headline scalar.
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
from scripts.eval_from_npz import build_methods  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def per_query_ndcg_and_confidence(probs, graded, qids, k=10):
    """Returns (qid -> {ndcg, max_prob, n_pool})."""
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
        ndcg = (dcg / idcg) if idcg > 0 else float("nan")
        out[q] = {"ndcg": ndcg, "max_prob": float(s.max()), "n_pool": len(idxs)}
    return out


def selective_curve(per_query: dict[str, dict], coverages):
    """Return (coverage -> mean NDCG@10 over the top-coverage-fraction
    of queries by confidence)."""
    items = [(d["max_prob"], d["ndcg"]) for d in per_query.values()
             if not np.isnan(d["ndcg"])]
    items.sort(key=lambda t: -t[0])  # confidence descending
    n = len(items)
    out = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        out[float(c)] = float(np.mean([t[1] for t in items[:k]]))
    return out, n


def auc_selective(curve):
    xs = sorted(curve.keys())
    ys = [curve[x] for x in xs]
    return float(np.trapz(ys, xs))


def run(npz_path: Path):
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

    signal_cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    methods = build_methods(signal_cols)

    coverages = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    out = {"n_queries": len(set(qids_test)), "methods": {}}
    for name, fusion in methods.items():
        try:
            fusion.fit(X[cal], y[cal], query_ids=qids_cal)
            p = fusion.fuse(X[test], query_ids=qids_test)
            pq = per_query_ndcg_and_confidence(p, graded[test], qids_test, k=10)
            curve, n_q = selective_curve(pq, coverages)
            out["methods"][name] = {
                "selective_ndcg_at_coverage": curve,
                "auc_selective_ndcg": auc_selective(curve),
                "ndcg_at_full_coverage": curve[1.00],
                "ndcg_at_coverage_0.50": curve[0.50],
                "ndcg_at_coverage_0.25": curve[0.25],
                "n_queries_with_label": len(pq),
            }
        except Exception as e:
            out["methods"][name] = {"error": repr(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--out", default="eval/selective_ndcg.json")
    ap.add_argument("--out-table", default="eval/selective_ndcg_table.txt")
    args = ap.parse_args()

    aggregate = {}
    for path in args.npz:
        name = Path(path).stem.replace("beir_", "").replace("_results", "")
        print(f"\n=== {name} ===")
        res = run(Path(path))
        aggregate[name] = res
        rows = []
        rows.append(f"{'method':<22} {'AUC':>6} {'NDCG@1.0':>9} {'NDCG@0.50':>10} {'NDCG@0.25':>10} {'NDCG@0.10':>10}")
        for m, ev in res["methods"].items():
            if "error" in ev:
                rows.append(f"{m:<22} ERROR")
                continue
            cov = ev["selective_ndcg_at_coverage"]
            rows.append(f"{m:<22} {ev['auc_selective_ndcg']:>6.3f} "
                        f"{cov[1.0]:>9.4f} {cov[0.5]:>10.4f} {cov[0.25]:>10.4f} {cov[0.1]:>10.4f}")
        print("\n".join(rows))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
