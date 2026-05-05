"""Primary evaluation on BEIR subsets.

This is the entry point the paper's headline numbers come from.
Run on an internet-connected machine (first run downloads the BEIR
subset; subsequent runs use the local cache). A GPU is required to
compute dense signals (BGE, E5, cross-encoder) at realistic scale;
the CPU-only fallback is for pipeline validation only.

Protocol
--------
For each named BEIR subset we:

1. Load the subset (queries, corpus, qrels) via
   :mod:`eval.beir_loader`.
2. Build per-query candidate pools of labelled positives + BM25-
   mined hard negatives.
3. Compute six raw signal scores per ``(query, passage)`` pair:
   BM25, BGE, E5, cross-encoder rerank, PPR on passage-similarity
   graph, MinHash/Jaccard.
4. Split queries 50/20/30 into calibration / validation / test.
5. Fit every fusion method on the calibration split; evaluate on
   test.
6. Report standard retrieval metrics (NDCG\\@10, MAP, Recall\\@100)
   alongside calibration metrics (ECE-15, ECE-10, Brier, NLL) and,
   where applicable, envelope coverage / sharpness.

Standard-methodology cross-reference
------------------------------------
NDCG\\@10 and Recall\\@100 are computed exactly as in the BEIR
paper (Thakur et~al., 2021). Binarisation of graded relevance for
ECE follows the BEIR convention ``rel >= 1``. Base BM25 uses the
Robertson--Zaragoza (2009) formulation with ``k1=1.5``, ``b=0.75``.
Dense encoders are run in their respective authors' recommended
configurations. The calibration / validation / test split is
query-level with fixed seed (``seed=2026``); we do not mine the
validation / test calibration set.

Usage
-----
```
PYTHONPATH=. python3 scripts/evaluate_beir.py \\
    --dataset scifact \\
    --out eval/beir_scifact_results.json
```
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

from eval.beir_loader import build_candidate_pool, load_beir, query_level_splits  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402
from src.fusion.calfuse_conformal import ConformalCalFuse  # noqa: E402
from src.fusion.calfuse_copula import CopulaCalFuse  # noqa: E402
from src.fusion.linear_learned import LinearLearnedFusion  # noqa: E402
from src.fusion.multicalibration import (  # noqa: E402
    Multicalibration,
    signal_dominance_subgroups,
    worst_subgroup_ece,
)
from src.fusion.reranker_fusion import RerankerFusion  # noqa: E402
from src.fusion.rrf import RRFFusion  # noqa: E402
from src.fusion.single_signal import SingleSignalFusion  # noqa: E402
from src.signals.cross_encoder import CrossEncoderSignal  # noqa: E402
from src.signals.dense_bge import DenseBGESignal  # noqa: E402
from src.signals.dense_e5 import DenseE5Signal  # noqa: E402
from src.signals.minhash_lsh import MinHashLSHSignal  # noqa: E402
from src.signals.ppr_graph import PPRGraphSignal  # noqa: E402


SIGNAL_ORDER_BASE = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]
SIGNAL_ORDER = list(SIGNAL_ORDER_BASE)  # mutated below if --include-monot5


def _compute_signal_matrix(pairs, subset, cpu_only: bool, include_monot5: bool = False):
    """Run the six production signals over the candidate pool, plus optional monoT5."""
    from src.signals.bm25 import BM25Signal

    # Fit BM25 once on the candidate-pool passages we actually score.
    corpus_texts = [p.passage_text for p in pairs]
    bm25 = BM25Signal().fit(corpus_texts)
    bge = DenseBGESignal(force_fallback=cpu_only)
    e5 = DenseE5Signal(force_fallback=cpu_only)
    ce = CrossEncoderSignal(force_fallback=cpu_only)

    # Sanity guard: when the caller explicitly asked for real signals (no
    # ``--cpu-only`` flag), verify each dense signal actually loaded its
    # published backbone. Silent fallback to deterministic hashed embeddings
    # has happened in practice on cloud VMs with mis-matched
    # ``transformers`` / ``sentence-transformers`` versions, and it produces
    # plausible-looking but non-comparable numbers.
    if not cpu_only:
        fell_back = [name for sig, name in [(bge, "bge"), (e5, "e5"), (ce, "cross_encoder")]
                     if sig._using_fallback]
        if fell_back:
            raise RuntimeError(
                f"Signals {fell_back} silently fell back to the hashed-embedding "
                f"surrogate despite --cpu-only not being set. Fix the underlying "
                f"sentence-transformers import (usually a missing tf-keras / Pillow "
                f"dep) before running BEIR eval — the numbers would not be "
                f"comparable to published baselines."
            )

    # PPR needs passage ids + texts; we use unique passages.
    unique_pids, idx = [], {}
    unique_texts = []
    for p in pairs:
        if p.passage_id not in idx:
            idx[p.passage_id] = len(unique_pids)
            unique_pids.append(p.passage_id)
            unique_texts.append(p.passage_text)
    ppr = PPRGraphSignal().fit_corpus(unique_pids, unique_texts)
    mh = MinHashLSHSignal()

    cols = [
        bm25.score_pairs(pairs),
        bge.score_pairs(pairs),
        e5.score_pairs(pairs),
        ce.score_pairs(pairs),
        ppr.score_pairs(pairs),
        mh.score_pairs(pairs),
    ]
    if include_monot5:
        from src.signals.monot5 import MonoT5Signal
        m5 = MonoT5Signal(force_fallback=cpu_only)
        cols.append(m5.score_pairs(pairs))
    return np.stack(cols, axis=1)


def _fit_and_evaluate(method_name, fusion, X_cal, y_cal, X_test, y_test,
                      qids_cal, qids_test, graded_test):
    fusion.fit(X_cal, y_cal, query_ids=qids_cal)
    p = fusion.fuse(X_test, query_ids=qids_test)
    ev = evaluate(p, y_test, graded_labels=graded_test, query_ids=qids_test,
                  include_reliability=False).as_dict()
    out = {"method": method_name, **ev}
    # If the method exposes a calibration envelope, record width too.
    if hasattr(fusion, "predict_envelope"):
        env = fusion.predict_envelope(X_test, query_ids=qids_test)
        out["envelope_mean_width"] = float(np.mean(env.p_hi - env.p_lo))
        out["envelope_coverage_at_0.5"] = float(
            np.mean(np.where(y_test == 1, env.p_hi >= 0.5, env.p_lo <= 0.5))
        )
    if hasattr(fusion, "report_") and hasattr(fusion.report_, "mode"):
        out["calfuse_mode"] = fusion.report_.mode
    return out


def main():
    parser = argparse.ArgumentParser(description="Primary CalFuse BEIR evaluation")
    parser.add_argument("--dataset", required=True,
                        help="BEIR subset name (nfcorpus, scifact, fiqa, trec-covid, arguana)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default eval/beir_{dataset}_results.json)")
    parser.add_argument("--top-k-bm25", type=int, default=100)
    parser.add_argument("--max-negatives", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-only", action="store_true",
                        help="Use offline deterministic fallbacks for dense/CE signals "
                        "(pipeline validation only; numbers not comparable to BEIR leaderboards)")
    parser.add_argument("--include-copula", action="store_true",
                        help="Include CopulaCalFuse in the methods dict. Off by default — "
                        "the parametric mode consistently outperforms it on our BEIR subsets. "
                        "Turn on for the dependence-regime ablation only.")
    parser.add_argument("--include-monot5", action="store_true",
                        help="Add the monoT5 reranker as a 7th signal column. Adds two "
                        "additional baseline rows (monoT5+Platt, monoT5_fusion) to the "
                        "method comparison.")
    args = parser.parse_args()
    if args.include_monot5 and "monot5" not in SIGNAL_ORDER:
        SIGNAL_ORDER.append("monot5")

    out_path = Path(args.out) if args.out else REPO / "eval" / f"beir_{args.dataset}_results.json"

    print(f"[1/5] Loading BEIR subset: {args.dataset}")
    subset = load_beir(args.dataset)
    print(f"      queries = {len(subset.queries)}   corpus = {len(subset.corpus)}")

    print(f"[2/5] Building candidate pool (top_k_bm25={args.top_k_bm25}, "
          f"max_negatives={args.max_negatives})")
    pairs, graded_list, qids, cand_k = build_candidate_pool(
        subset,
        top_k_bm25=args.top_k_bm25,
        max_negatives_per_query=args.max_negatives,
        seed=args.seed,
    )
    print(f"      pairs   = {len(pairs)}   mean pool = {np.mean(cand_k):.1f}")

    graded = np.array(graded_list, dtype=np.float64)
    y = (graded >= 1.0).astype(np.int64)

    print(f"[3/5] Computing signal matrix ({'CPU-only' if args.cpu_only else 'GPU if available'})")
    X = _compute_signal_matrix(pairs, subset, cpu_only=args.cpu_only,
                               include_monot5=args.include_monot5)

    print(f"[4/5] Query-level split (seed={args.seed})")
    splits = query_level_splits(qids, seed=args.seed)
    split_arr = np.array([splits[q] for q in qids])
    cal_mask = split_arr == "calibration"
    test_mask = split_arr == "test"
    qids_cal = [qids[i] for i in range(len(qids)) if cal_mask[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test_mask[i]]

    X_cal, y_cal = X[cal_mask], y[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    graded_test = graded[test_mask]

    print(f"      calibration = {cal_mask.sum()}   test = {test_mask.sum()}   "
          f"base rate (test) = {y_test.mean():.3f}")

    print("[5/5] Fitting fusion methods")
    ce_col = SIGNAL_ORDER.index("cross_encoder")
    bm25_col = SIGNAL_ORDER.index("bm25")
    bge_col = SIGNAL_ORDER.index("dense_bge")
    e5_col = SIGNAL_ORDER.index("dense_e5")
    monot5_col = SIGNAL_ORDER.index("monot5") if "monot5" in SIGNAL_ORDER else None
    methods = {
        # SOTA single-retriever baselines (each is a standard BEIR reference
        # point, calibrated with Platt so the ECE comparison is well-defined).
        "bm25_platt": SingleSignalFusion(bm25_col, name="bm25_platt"),
        "bge_platt": SingleSignalFusion(bge_col, name="bge_platt"),
        "e5_platt": SingleSignalFusion(e5_col, name="e5_platt"),
        "cross_encoder_platt": SingleSignalFusion(ce_col, name="cross_encoder_platt"),
        # Classical and learned fusion baselines.
        "rrf": RRFFusion(),
        "linear_learned": LinearLearnedFusion(),
        "reranker_fusion": RerankerFusion(reranker_col=ce_col),
        # CalFuse variants under test.
        "calfuse_parametric": CalFuseFusion(force_mode="parametric"),
        "calfuse_multical": Multicalibration(
            base=CalFuseFusion(force_mode="parametric"),
            subgroup_fn=signal_dominance_subgroups(),
        ),
        "calfuse_conformal": ConformalCalFuse(
            base=CalFuseFusion(force_mode="parametric"),
            subgroup_fn=signal_dominance_subgroups(),
        ),
    }
    if args.include_copula:
        methods["calfuse_copula"] = CopulaCalFuse(shrinkage=0.2)
    if monot5_col is not None:
        methods["monot5_platt"] = SingleSignalFusion(monot5_col, name="monot5_platt")
        methods["monot5_reranker_fusion"] = RerankerFusion(reranker_col=monot5_col)

    results = []
    for name, fusion in methods.items():
        print(f"  - {name}")
        try:
            results.append(_fit_and_evaluate(
                name, fusion, X_cal, y_cal, X_test, y_test,
                qids_cal, qids_test, graded_test,
            ))
        except Exception as e:
            results.append({"method": name, "error": repr(e)})

    # Worst-subgroup ECE for every method (signal-family dominance strata).
    M = np.asarray(signal_dominance_subgroups()(X_test, qids_test), dtype=bool)
    for r in results:
        if "error" in r:
            continue
        # Re-fuse to get probabilities (we did not cache).
        fusion = methods[r["method"]]
        p = fusion.fuse(X_test, query_ids=qids_test)
        r["worst_subgroup_ece_15"] = worst_subgroup_ece(p, y_test, M, n_bins=15, n_min=25)

    out = {
        "_type": "calfuse_beir_results",
        "dataset": args.dataset,
        "seed": args.seed,
        "n_queries_total": len(subset.queries),
        "n_pairs_total": int(len(pairs)),
        "n_pairs_calibration": int(cal_mask.sum()),
        "n_pairs_test": int(test_mask.sum()),
        "test_positive_rate": float(y_test.mean()),
        "mean_candidate_pool_size": float(np.mean(cand_k)),
        "signals": SIGNAL_ORDER,
        "methods": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nWrote {out_path}")

    # Also save the raw signal matrix + labels so downstream ablations
    # (calibrator swap, threshold sweep, subgroup breakdown) can be re-run
    # in seconds without paying the signal-computation cost again.
    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        graded=graded,
        qids=np.array(qids, dtype=object),
        split=split_arr,
        signal_order=np.array(SIGNAL_ORDER, dtype=object),
    )
    print(f"Wrote {npz_path}")

    print("\nHeadline table (ECE_15 / NDCG@10 / worst-subgroup ECE_15):")
    print(f"{'method':<24} {'ECE_15':>8} {'NDCG@10':>8} {'worst_ECE_15':>14}")
    for r in results:
        if "error" in r:
            print(f"{r['method']:<24} ERROR: {r['error']}")
            continue
        print(f"{r['method']:<24} "
              f"{r['ece_15']:>8.4f} "
              f"{r.get('ndcg_10') or 0.0:>8.4f} "
              f"{r.get('worst_subgroup_ece_15', float('nan')):>14.4f}")


if __name__ == "__main__":
    main()
