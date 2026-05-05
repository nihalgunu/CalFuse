"""Per-method top-5 composition: number of positives, position
diversity (which positions get filled by positives). Run after
abstention_and_rank_analysis to triangulate the scifact mechanism.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sst

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.beir_loader import build_candidate_pool, load_beir, query_level_splits  # noqa: E402
from scripts.eval_from_npz import build_methods  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]
SUBSETS = ["nfcorpus", "scifact", "fiqa", "arguana", "scidocs"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
METHODS = ["bm25_platt", "bge_platt", "linear_learned",
           "calfuse_parametric", "calfuse_conformal"]
TOP_K = 5


def main():
    cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    summary = {}
    for subset in SUBSETS:
        sub = load_beir(subset)
        npz = np.load(REPO / f"eval/beir_{subset}_results.npz", allow_pickle=True)
        X = npz["X"].astype(np.float64)
        y = npz["y"].astype(np.int64)
        qids = list(npz["qids"])
        pairs, _, qids_built, _ = build_candidate_pool(
            sub, top_k_bm25=100, max_negatives_per_query=40, seed=2026)
        assert qids == qids_built
        splits = query_level_splits(qids, seed=2026)
        split_arr = np.array([splits[q] for q in qids])
        test_mask = split_arr == "test"
        cal_mask = split_arr == "calibration"
        qids_cal = [qids[i] for i in range(len(qids)) if cal_mask[i]]
        qids_test = [qids[i] for i in range(len(qids)) if test_mask[i]]
        y_test = y[test_mask]

        # Per-method top-5 stats per query.
        method_stats = {}
        methods = build_methods(cols)
        for m in METHODS:
            f = methods[m]
            f.fit(X[cal_mask], y[cal_mask], query_ids=qids_cal)
            probs = f.fuse(X[test_mask], query_ids=qids_test)
            by_q = {}
            for i, q in enumerate(qids_test):
                by_q.setdefault(q, []).append(i)
            stats = {}
            for q, idxs in by_q.items():
                idxs = np.asarray(idxs)
                p = probs[idxs]
                order = np.argsort(-p, kind="mergesort")[:TOP_K]
                is_pos = [int(y_test[idxs[j]] == 1) for j in order]
                stats[q] = dict(n_pos_in_top5=int(sum(is_pos)),
                                pos_positions=tuple(i + 1 for i, p_ in enumerate(is_pos) if p_),
                                top5_pids=tuple(int(idxs[j]) for j in order))
            method_stats[m] = stats

        # Aggregate across queries from the union of seed-50-query
        # samples (use seed=2026 list as proxy for "queries the LLM
        # actually saw"; since the candidate pool is the same, the
        # top-5 doesn't depend on seed).
        # Simpler: aggregate over ALL test queries.
        agg = {}
        for m in METHODS:
            n_pos_arr = np.array([s["n_pos_in_top5"]
                                  for s in method_stats[m].values()])
            agg[m] = dict(
                mean_n_pos_in_top5=float(n_pos_arr.mean()),
                std_n_pos_in_top5=float(n_pos_arr.std(ddof=1)) if len(n_pos_arr) > 1 else 0.0,
                pct_with_pos_in_top5=float((n_pos_arr > 0).mean() * 100),
                pct_with_2_or_more_pos=float((n_pos_arr >= 2).mean() * 100),
                pct_with_3_or_more_pos=float((n_pos_arr >= 3).mean() * 100),
                n_test_queries=int(len(n_pos_arr)),
            )

        # Pairwise overlap between Linear-Learned and CalFuse-Parametric top-5.
        ll = method_stats["linear_learned"]
        cf = method_stats["calfuse_parametric"]
        common = sorted(set(ll) & set(cf))
        # Jaccard of the top-5 pid sets per query.
        jaccards = []
        identical_pos = 0
        for q in common:
            a = set(ll[q]["top5_pids"])
            b = set(cf[q]["top5_pids"])
            j = len(a & b) / max(1, len(a | b))
            jaccards.append(j)
            if ll[q]["n_pos_in_top5"] == cf[q]["n_pos_in_top5"]:
                identical_pos += 1
        agg["_overlap_LL_vs_CalFuseP"] = dict(
            mean_jaccard_top5=float(np.mean(jaccards)),
            pct_identical_n_pos=float(100 * identical_pos / max(1, len(common))),
            n_queries=len(common),
        )

        # Difference in n_pos_in_top5 per query (Linear vs CalFuse).
        diffs = np.array([cf[q]["n_pos_in_top5"] - ll[q]["n_pos_in_top5"]
                          for q in common], dtype=float)
        if len(diffs) >= 2:
            t, p = sst.ttest_1samp(diffs, 0)
            agg["_diff_n_pos_LL_vs_CalFuseP"] = dict(
                mean_diff=float(diffs.mean()),
                std_diff=float(diffs.std(ddof=1)),
                t=float(t), p=float(p),
                n_queries=int(len(diffs)),
            )

        summary[subset] = agg

        print(f"\n=== {subset} ===")
        print(f"  {'method':<22} {'mean_n_pos':>11} {'pct_any':>8} "
              f"{'pct≥2':>7} {'pct≥3':>7}  (n_test={agg['linear_learned']['n_test_queries']})")
        for m in METHODS:
            a = agg[m]
            print(f"  {m:<22} {a['mean_n_pos_in_top5']:>11.3f} "
                  f"{a['pct_with_pos_in_top5']:>7.1f}% "
                  f"{a['pct_with_2_or_more_pos']:>6.1f}% "
                  f"{a['pct_with_3_or_more_pos']:>6.1f}%")
        ov = agg["_overlap_LL_vs_CalFuseP"]
        print(f"  Linear-Learned vs CalFuse-Parametric:")
        print(f"    mean Jaccard(top-5 pids) = {ov['mean_jaccard_top5']:.3f}")
        print(f"    same n_pos in top-5     = {ov['pct_identical_n_pos']:.1f}% of queries")
        d = agg["_diff_n_pos_LL_vs_CalFuseP"]
        print(f"    n_pos diff (CF-LL)      = {d['mean_diff']:+.4f} ± {d['std_diff']:.3f}  t={d['t']:.2f} p={d['p']:.3f}")

    out = REPO / "eval/topk_composition.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
