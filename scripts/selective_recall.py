"""Selective Recall@k vs query coverage --- the hallucination-proxy
operating curve.

A retrieval-augmented LM only hallucinates when the retrieved
passages do not actually contain the answer. The simplest
hallucination proxy that does not require a generation step is
``did the top-k retrieved passages include any labelled positive''.
A system with a calibrated abstention rule will refuse to answer
queries whose retrieval is unlikely to contain a positive --- and
therefore avoid the hallucinations that would otherwise follow.

We mirror the selective-NDCG protocol: sort queries by their
maximum fused probability, answer the top c-fraction, and report
mean Recall@10 over the answered queries. As c shrinks, a calibrated
fusion rule keeps the queries it can answer correctly; recall@10
should rise toward 1.

Together with selective NDCG\@10 (Section~\ref{sec:selective-ndcg})
this gives the two metrics a downstream RAG system actually cares
about: the answer is present (recall) and ranked high (NDCG).
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

from scripts.eval_from_npz import build_methods  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def per_query_recall_and_confidence(probs, y, qids, k=10):
    by_q = {}
    for i, q in enumerate(qids):
        by_q.setdefault(q, []).append(i)
    out = {}
    for q, idxs in by_q.items():
        idxs = np.array(idxs, dtype=np.int64)
        s = probs[idxs]
        rel = y[idxs]
        n_pos = int(rel.sum())
        if n_pos == 0:
            continue
        order = np.argsort(-s, kind="mergesort")[:k]
        retrieved_pos = int(rel[order].sum())
        out[q] = {
            "recall_at_k": retrieved_pos / n_pos,
            "any_pos_in_top_k": float(retrieved_pos > 0),
            "max_prob": float(s.max()),
            "n_pool": len(idxs),
            "n_pos": n_pos,
        }
    return out


def selective_curve(per_query, coverages, metric="recall_at_k"):
    items = [(d["max_prob"], d[metric]) for d in per_query.values()]
    items.sort(key=lambda t: -t[0])
    n = len(items)
    out = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        out[float(c)] = float(np.mean([t[1] for t in items[:k]]))
    return out


def run(npz_path: Path, k=10):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    qids = list(data["qids"])
    split = np.asarray(data["split"])
    cal = split == "calibration"
    test = split == "test"
    qids_cal = [qids[i] for i in range(len(qids)) if cal[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test[i]]

    signal_cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    methods = build_methods(signal_cols)

    coverages = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    out = {"methods": {}}
    for name, fusion in methods.items():
        try:
            fusion.fit(X[cal], y[cal], query_ids=qids_cal)
            p = fusion.fuse(X[test], query_ids=qids_test)
            pq = per_query_recall_and_confidence(p, y[test], qids_test, k=k)
            recall_curve = selective_curve(pq, coverages, "recall_at_k")
            hit_curve = selective_curve(pq, coverages, "any_pos_in_top_k")
            # Hallucination rate = 1 - any_pos_in_top_k (no positive => LM has to make stuff up).
            hallu = {c: 1.0 - v for c, v in hit_curve.items()}
            out["methods"][name] = {
                "n_queries_with_positive": len(pq),
                "recall_at_k": recall_curve,
                "hit_at_k": hit_curve,
                "hallucination_proxy_at_coverage": hallu,
            }
        except Exception as e:
            out["methods"][name] = {"error": repr(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="eval/selective_recall.json")
    args = ap.parse_args()

    aggregate = {}
    for path in args.npz:
        name = Path(path).stem.replace("beir_", "").replace("_results", "")
        print(f"\n=== {name} ===")
        res = run(Path(path), k=args.k)
        aggregate[name] = res
        print(f"{'method':<22} {'rec@1.0':>8} {'rec@0.5':>8} {'rec@0.25':>9} {'rec@0.1':>8} {'hallu@0.5':>10}")
        for m, ev in res["methods"].items():
            if "error" in ev:
                continue
            r = ev["recall_at_k"]; h = ev["hallucination_proxy_at_coverage"]
            print(f"{m:<22} {r[1.0]:>8.4f} {r[0.5]:>8.4f} {r[0.25]:>9.4f} {r[0.1]:>8.4f} {h[0.5]:>10.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
