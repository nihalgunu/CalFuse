"""Two follow-up offline analyses computed purely from cached data:

1. **Selective abstention curve**: at coverage c, rank queries by
   each method's max top-1 fused probability and answer only the
   most-confident c-fraction. Report hallu_NR (hallucination among
   non-refused, among answered queries) at c in {0.1, 0.25, 0.5,
   0.75, 1.0}, multi-seed mean ± std. The system-level question:
   does CalFuse's calibration deliver a real abstention policy that
   reduces hallucination at fixed coverage on subsets beyond just
   scifact?

2. **Mean-rank-of-first-positive (mechanistic test)**: for each
   query, what is the rank (1..5, or 6 if absent) of the first
   labelled positive in the top-5 retrieved? Lower = pushes
   positives higher. If CalFuse beats Linear-Learned on this on
   scifact, that is the *mechanism* behind the hallucination
   reduction (the LLM sees the right passage earlier).

Inputs:
- eval/beir_<subset>_results.npz  (cached signal scores)
- eval/multiseed/llm_hallu_<subset>_seed<seed>.json (verdicts)

Outputs:
- eval/selective_abstention_multiseed.json
- eval/mean_rank_positive_multiseed.json
- prints summary tables.

No LLM calls. Run locally; takes ~30s.
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
COVERAGES = [0.10, 0.25, 0.50, 0.75, 1.00]
TOP_K = 5
N_QUERIES = 50


def load_cached(subset):
    """Return cached signals + alignment with re-built candidate pool."""
    npz = np.load(REPO / f"eval/beir_{subset}_results.npz", allow_pickle=True)
    X = npz["X"].astype(np.float64)
    y = npz["y"].astype(np.int64)
    qids = list(npz["qids"])
    split = np.asarray(npz["split"])

    sub = load_beir(subset)
    pairs, _, qids_built, _ = build_candidate_pool(
        sub, top_k_bm25=100, max_negatives_per_query=40, seed=2026)
    splits = query_level_splits(qids_built, seed=2026)
    split_arr = np.array([splits[q] for q in qids_built])
    test_mask = split_arr == "test"
    cal_mask = split_arr == "calibration"

    # Sanity: cached qids must equal re-built qids per split.
    assert qids == qids_built, f"{subset}: qid alignment failed"

    pair_pids = [p.passage_id for p in pairs]
    return dict(X=X, y=y, qids=qids, split=split,
                test_mask=test_mask, cal_mask=cal_mask,
                pair_pids=pair_pids)


def fit_predict(cache, method_name):
    cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    methods = build_methods(cols)
    f = methods[method_name]
    qids_cal = [cache["qids"][i] for i in range(len(cache["qids"]))
                if cache["cal_mask"][i]]
    qids_test = [cache["qids"][i] for i in range(len(cache["qids"]))
                 if cache["test_mask"][i]]
    f.fit(cache["X"][cache["cal_mask"]], cache["y"][cache["cal_mask"]],
          query_ids=qids_cal)
    probs = f.fuse(cache["X"][cache["test_mask"]], query_ids=qids_test)
    return probs, qids_test


def top5_per_query(probs, qids_test, y_test):
    """Group test pairs by qid, return per-query (top1_prob,
    rank_of_first_pos_in_top5, n_pos_in_top5).

    rank_of_first_pos_in_top5 = position (1..5) of the first labelled
    positive among the top-5 retrieved; 6 if none.
    """
    by_q = {}
    for i, q in enumerate(qids_test):
        by_q.setdefault(q, []).append(i)
    out = {}
    for q, idxs in by_q.items():
        idxs = np.asarray(idxs)
        p = probs[idxs]
        order = np.argsort(-p, kind="mergesort")[:TOP_K]
        top1_prob = float(p[order[0]]) if len(order) else 0.0
        is_pos = [int(y_test[idxs[j]] == 1) for j in order]
        if any(is_pos):
            first_rank = is_pos.index(1) + 1
        else:
            first_rank = TOP_K + 1
        out[q] = dict(top1_prob=top1_prob,
                      rank_of_first_pos=first_rank,
                      n_pos_in_top5=int(sum(is_pos)),
                      any_pos_in_top5=int(any(is_pos)))
    return out


def selective_hallu_nr(rows, coverages):
    """rows: list of (top1_prob, verdict_in {grounded, on_retrieval, fabricated, refused}).
    Returns dict {coverage: hallu_NR_pct}."""
    if not rows: return {c: float("nan") for c in coverages}
    rows = sorted(rows, key=lambda r: -r[0])
    n = len(rows)
    out = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        sub = rows[:k]
        n_grd = sum(1 for _, v in sub if v == "grounded")
        n_oR = sum(1 for _, v in sub if v == "on_retrieval")
        n_fab = sum(1 for _, v in sub if v == "fabricated")
        n_NR = n_grd + n_oR + n_fab
        out[c] = float("nan") if n_NR == 0 else 100.0 * (n_oR + n_fab) / n_NR
    return out


def main():
    selective_results = {}
    rank_results = {}

    for subset in SUBSETS:
        print(f"\n=== {subset} ===")
        cache = load_cached(subset)
        y_test = cache["y"][cache["test_mask"]]

        # Fit each method ONCE per subset (calibrator state is
        # deterministic given fixed seed=2026 cal split).
        method_topk = {}
        for m in METHODS:
            probs, qids_test = fit_predict(cache, m)
            method_topk[m] = (top5_per_query(probs, qids_test, y_test),
                              qids_test)

        sel_subset = {m: {c: [] for c in COVERAGES} for m in METHODS}
        rank_subset = {m: {"mean_rank": [], "mrr5": [], "p_at_1": [],
                            "p_at_3": [], "p_at_5": [], "any_pos_at_5": []}
                       for m in METHODS}

        for seed in SEEDS:
            vfile = REPO / f"eval/multiseed/llm_hallu_{subset}_seed{seed}.json"
            if not vfile.exists():
                print(f"  miss verdicts: {vfile.name}")
                continue
            d = json.load(open(vfile))
            for m in METHODS:
                ev = d["methods"].get(m, {})
                if not ev or "examples" not in ev: continue
                examples = ev["examples"]
                top, _ = method_topk[m]

                # Selective abstention rows.
                rows = []
                for ex in examples:
                    q = ex["qid"]
                    v = ex["verdict"]
                    if q not in top: continue
                    rows.append((top[q]["top1_prob"], v))
                sel_at = selective_hallu_nr(rows, COVERAGES)
                for c, h in sel_at.items():
                    if not np.isnan(h): sel_subset[m][c].append(h)

                # Rank mechanistic — across the 50 sampled queries.
                ranks = [top[ex["qid"]]["rank_of_first_pos"]
                         for ex in examples if ex["qid"] in top]
                if not ranks: continue
                ranks = np.asarray(ranks, dtype=float)
                # mean rank: 1..5 if pos in top5, 6 if absent (penalty)
                rank_subset[m]["mean_rank"].append(float(ranks.mean()))
                # MRR@5: 1/rank if in top5, 0 if absent
                mrr = np.where(ranks <= 5, 1.0 / np.maximum(ranks, 1.0), 0.0)
                rank_subset[m]["mrr5"].append(float(mrr.mean()))
                rank_subset[m]["p_at_1"].append(float((ranks == 1).mean()))
                rank_subset[m]["p_at_3"].append(float((ranks <= 3).mean()))
                rank_subset[m]["p_at_5"].append(float((ranks <= 5).mean()))
                rank_subset[m]["any_pos_at_5"].append(float((ranks <= 5).mean()))

        # Aggregate selective.
        sel_agg = {}
        for m in METHODS:
            sel_agg[m] = {}
            for c in COVERAGES:
                arr = np.array(sel_subset[m][c], dtype=float)
                if len(arr) == 0:
                    sel_agg[m][c] = dict(mean=float("nan"), std=float("nan"),
                                         n=0)
                else:
                    sel_agg[m][c] = dict(
                        mean=float(arr.mean()),
                        std=float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                        n=int(len(arr)))
        selective_results[subset] = sel_agg

        # Aggregate rank.
        rank_agg = {}
        for m in METHODS:
            rank_agg[m] = {}
            for k, v in rank_subset[m].items():
                arr = np.array(v, dtype=float)
                if len(arr) == 0:
                    rank_agg[m][k] = dict(mean=float("nan"),
                                          std=float("nan"), n=0)
                else:
                    rank_agg[m][k] = dict(
                        mean=float(arr.mean()),
                        std=float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                        n=int(len(arr)))
        rank_results[subset] = rank_agg

        # Print quick summary per subset.
        print(f"  selective hallu_NR (mean across 5 seeds), {subset}:")
        print(f"  {'method':<22} " + " ".join(f"c={c:>4.2f}" for c in COVERAGES))
        for m in METHODS:
            row = " ".join(f"{sel_agg[m][c]['mean']:>5.1f}"
                           for c in COVERAGES)
            print(f"  {m:<22} {row}")
        print(f"  mean rank-of-first-pos (1=best, 6=absent):")
        for m in METHODS:
            r = rank_agg[m]["mean_rank"]
            mrr = rank_agg[m]["mrr5"]
            p1 = rank_agg[m]["p_at_1"]
            print(f"  {m:<22}  mean_rank={r['mean']:.3f}±{r['std']:.3f}  "
                  f"MRR@5={mrr['mean']:.3f}±{mrr['std']:.3f}  "
                  f"P@1={p1['mean']*100:.1f}±{p1['std']*100:.1f}%")

        # Paired-t: CalFuse-Parametric vs Linear-Learned on selective + rank.
        for c in COVERAGES:
            a = np.array(sel_subset["calfuse_parametric"][c])
            b = np.array(sel_subset["linear_learned"][c])
            if len(a) >= 2 and len(a) == len(b):
                t, p = sst.ttest_rel(a, b)
                d = float(np.mean(a - b))
                if p < 0.10:
                    print(f"  [paired-t] selective@c={c:.2f}  CalFuse-P "
                          f"vs Linear-Learned: Δ={d:+.2f}pp  t={t:.2f} p={p:.3f}")

        for k in ["mean_rank", "mrr5", "p_at_5"]:
            a = np.array(rank_subset["calfuse_parametric"][k])
            b = np.array(rank_subset["linear_learned"][k])
            if len(a) >= 2 and len(a) == len(b):
                t, p = sst.ttest_rel(a, b)
                d = float(np.mean(a - b))
                if p < 0.10:
                    print(f"  [paired-t] {k:<12} CalFuse-P vs Linear-Learned: "
                          f"Δ={d:+.4f}  t={t:.2f} p={p:.3f}")

    # Write JSON outputs.
    out_sel = REPO / "eval/selective_abstention_multiseed.json"
    json.dump(selective_results, open(out_sel, "w"), indent=2)
    print(f"\nWrote {out_sel}")
    out_rank = REPO / "eval/mean_rank_positive_multiseed.json"
    json.dump(rank_results, open(out_rank, "w"), indent=2)
    print(f"Wrote {out_rank}")


if __name__ == "__main__":
    main()
