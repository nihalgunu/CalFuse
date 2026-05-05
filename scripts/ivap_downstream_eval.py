"""End-to-end downstream eval: does the IVAP envelope outperform the
point estimate as an abstention signal in the LLM-grounding pipeline?

Setup. For each (subset, seed) we already have:
  - cached signal scores in eval/beir_<sub>_results.npz
  - 50 sampled queries with LLM verdicts in eval/multiseed/...

Procedure (no LLM re-run; pure offline analysis):
  1. Fit CalFuse-Parametric on the calibration split (point estimates).
  2. Fit a FastVennAbersPredictor on (cal point estimate, cal labels).
  3. For each test pair, derive (p_lo, p_hi). For each query, take the
     max top-1 fused passage's (p_lo, p_hi, midpoint, width).
  4. Build selective abstention curves using four ranking signals:
       (a) point_top1   = baseline (CalFuse-Parametric point estimate)
       (b) ivap_lo_top1 = IVAP lower envelope at top-1 (more conservative)
       (c) ivap_mid_top1 = IVAP midpoint at top-1 (mean of envelope)
       (d) ivap_lo_minus_alpha_width = ivap_lo_top1 - alpha * width
                                       (penalises wide envelopes)
  5. Report hallu_NR @ c in {0.10, 0.25, 0.50, 0.75, 1.00}, 5-seed
     mean ± std, paired-t against the point-estimate baseline.

Output: eval/ivap_downstream_eval.json + summary table on stdout.
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
from src.conformal.venn_abers import FastVennAbersPredictor  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]
SUBSETS = ["nfcorpus", "scifact", "fiqa", "arguana", "scidocs"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
COVERAGES = [0.10, 0.25, 0.50, 0.75, 1.00]
ALPHA_WIDTH = 0.5  # penalty on envelope width
TOP_K = 5


def per_query_signals(probs, p_lo, p_hi, qids_test, y_test):
    """Return per-query top-1 signals for each ranking criterion."""
    by_q = {}
    for i, q in enumerate(qids_test):
        by_q.setdefault(q, []).append(i)
    out = {}
    for q, idxs in by_q.items():
        idxs = np.asarray(idxs)
        s = probs[idxs]
        order = np.argsort(-s, kind="mergesort")[:TOP_K]
        # Top-1 signals = signals on the highest-point-estimate passage
        # (so all four ranking criteria agree on which passage is the
        # candidate; they only disagree on how confident we are).
        i1 = idxs[order[0]]
        out[q] = dict(
            point=float(probs[i1]),
            ivap_lo=float(p_lo[i1]),
            ivap_hi=float(p_hi[i1]),
            ivap_mid=0.5 * (float(p_lo[i1]) + float(p_hi[i1])),
            ivap_width=float(p_hi[i1] - p_lo[i1]),
        )
    return out


def selective_hallu(rows, coverages):
    """rows: list of (score, verdict). Sort by -score, take top-c
    fraction, return hallu_NR%.
    """
    if not rows: return {c: float("nan") for c in coverages}
    rows = sorted(rows, key=lambda r: -r[0])
    n = len(rows)
    out = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        sub = rows[:k]
        n_oR = sum(1 for _, v in sub if v == "on_retrieval")
        n_fab = sum(1 for _, v in sub if v == "fabricated")
        n_grd = sum(1 for _, v in sub if v == "grounded")
        n_NR = n_oR + n_fab + n_grd
        out[c] = float("nan") if n_NR == 0 else 100.0 * (n_oR + n_fab) / n_NR
    return out


def process_subset(subset):
    print(f"\n=== {subset} ===")
    sub = load_beir(subset)
    pairs, _, qids_built, _ = build_candidate_pool(
        sub, top_k_bm25=100, max_negatives_per_query=40, seed=2026)
    npz = np.load(REPO / f"eval/beir_{subset}_results.npz",
                  allow_pickle=True)
    qids = list(npz["qids"]); assert qids == qids_built
    X = npz["X"].astype(np.float64); y = npz["y"].astype(np.int64)
    splits = query_level_splits(qids, seed=2026)
    sa = np.array([splits[q] for q in qids])
    test, cal = sa == "test", sa == "calibration"
    qids_cal = [qids[i] for i in range(len(qids)) if cal[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test[i]]

    # 1. Fit CalFuse-Parametric on cal split.
    cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    methods = build_methods(cols)
    f = methods["calfuse_parametric"]
    f.fit(X[cal], y[cal], query_ids=qids_cal)
    cal_probs = f.fuse(X[cal], query_ids=qids_cal)
    test_probs = f.fuse(X[test], query_ids=qids_test)

    # 2. IVAP on (cal point estimates, cal labels).
    ivap = FastVennAbersPredictor()
    ivap.fit(cal_probs, y[cal])
    env = ivap.predict_envelope(test_probs)

    # 3. Per-query top-1 signals.
    sig_per_q = per_query_signals(
        test_probs, env.p_lo, env.p_hi, qids_test, y[test])
    print(f"  envelope width on test: mean={env.width.mean():.4f} "
          f"med={np.median(env.width):.4f} "
          f"max={env.width.max():.4f}")

    # 4. Selective curves per seed.
    ranking_keys = ["point", "ivap_lo", "ivap_mid"]
    # Combined: ivap_lo - alpha * width.
    def combined_score(s):
        return s["ivap_lo"] - ALPHA_WIDTH * s["ivap_width"]

    per_seed = {k: {c: [] for c in COVERAGES} for k in ranking_keys + ["combined"]}
    width_per_q_per_seed = []
    width_grouped = {"grounded": [], "on_retrieval": [],
                      "fabricated": [], "refused": []}

    for seed in SEEDS:
        vfile = REPO / f"eval/multiseed/llm_hallu_{subset}_seed{seed}.json"
        if not vfile.exists(): continue
        d = json.load(open(vfile))
        examples = d["methods"]["calfuse_parametric"]["examples"]
        for k in ranking_keys:
            rows = []
            for ex in examples:
                q = ex["qid"]
                if q not in sig_per_q: continue
                rows.append((sig_per_q[q][k], ex["verdict"]))
            sel = selective_hallu(rows, COVERAGES)
            for c, h in sel.items():
                if not np.isnan(h): per_seed[k][c].append(h)
        # Combined.
        rows = []
        for ex in examples:
            q = ex["qid"]
            if q not in sig_per_q: continue
            rows.append((combined_score(sig_per_q[q]), ex["verdict"]))
        sel = selective_hallu(rows, COVERAGES)
        for c, h in sel.items():
            if not np.isnan(h): per_seed["combined"][c].append(h)

        # Width vs verdict (orthogonal diagnostic).
        for ex in examples:
            q = ex["qid"]
            if q not in sig_per_q: continue
            width_grouped[ex["verdict"]].append(sig_per_q[q]["ivap_width"])

    summary = {}
    for k in ranking_keys + ["combined"]:
        summary[k] = {}
        for c in COVERAGES:
            arr = np.array(per_seed[k][c], dtype=float)
            if len(arr) == 0:
                summary[k][c] = dict(mean=float("nan"), std=float("nan"),
                                     n=0, raw=[])
            else:
                summary[k][c] = dict(mean=float(arr.mean()),
                                      std=float(arr.std(ddof=1)
                                                if len(arr) > 1 else 0),
                                      n=int(len(arr)),
                                      raw=arr.tolist())

    # Paired-t: each IVAP variant vs point.
    paired = {}
    for k in ["ivap_lo", "ivap_mid", "combined"]:
        paired[k] = {}
        for c in COVERAGES:
            a = np.array(per_seed[k][c])
            b = np.array(per_seed["point"][c])
            if len(a) >= 2 and len(a) == len(b):
                t, p = sst.ttest_rel(a, b)
                paired[k][c] = dict(diff_mean=float((a - b).mean()),
                                    t=float(t), p=float(p),
                                    n=int(len(a)))

    width_stats = {v: dict(mean=float(np.mean(width_grouped[v])) if width_grouped[v] else float("nan"),
                            std=float(np.std(width_grouped[v], ddof=1))
                                  if len(width_grouped[v]) > 1 else 0.0,
                            n=len(width_grouped[v]))
                   for v in width_grouped}

    print(f"  selective hallu_NR (mean across 5 seeds), {subset}:")
    print(f"  {'rank-by':<12} " + " ".join(f"c={c:>4.2f}" for c in COVERAGES))
    for k in ranking_keys + ["combined"]:
        row = " ".join(f"{summary[k][c]['mean']:>5.1f}" for c in COVERAGES)
        print(f"  {k:<12} {row}")
    print(f"  paired-t vs point (Δ pp, p):")
    for k in ["ivap_lo", "ivap_mid", "combined"]:
        line = []
        for c in COVERAGES:
            r = paired[k].get(c)
            if not r: continue
            mark = " *" if r["p"] < 0.05 else ""
            line.append(f"c={c:.2f}: Δ={r['diff_mean']:+.2f}pp p={r['p']:.3f}{mark}")
        print(f"  {k:<12}  " + " ".join(line))
    print(f"  envelope width by verdict:")
    for v in ["grounded", "on_retrieval", "fabricated", "refused"]:
        s = width_stats[v]
        if s["n"]:
            print(f"    {v:<14} n={s['n']:<4d} width={s['mean']:.4f}±{s['std']:.4f}")

    return dict(summary=summary, paired=paired,
                width_by_verdict=width_stats,
                env_width_mean=float(env.width.mean()),
                env_width_median=float(np.median(env.width)))


def main():
    out = {}
    for subset in SUBSETS:
        out[subset] = process_subset(subset)
    Path(REPO / "eval/ivap_downstream_eval.json").write_text(
        json.dumps(out, indent=1))
    print(f"\nWrote eval/ivap_downstream_eval.json")

    # Final cross-subset summary at c=0.25 (the key selective regime).
    print("\n=== Cross-subset summary @ c=0.25, 5-seed mean hallu_NR (%) ===")
    print(f"{'subset':<10} {'point':>7} {'ivap_lo':>8} {'ivap_mid':>9} {'combined':>9}  best vs point")
    for subset in SUBSETS:
        s = out[subset]["summary"]
        bestk = min(["ivap_lo", "ivap_mid", "combined"],
                    key=lambda k: s[k][0.25]["mean"])
        diff = s[bestk][0.25]["mean"] - s["point"][0.25]["mean"]
        p = out[subset]["paired"][bestk].get(0.25, {}).get("p", float("nan"))
        marker = " ✓" if (diff < 0 and p < 0.10) else (" -" if diff < 0 else " ✗")
        print(f"{subset:<10} {s['point'][0.25]['mean']:>7.1f} "
              f"{s['ivap_lo'][0.25]['mean']:>8.1f} "
              f"{s['ivap_mid'][0.25]['mean']:>9.1f} "
              f"{s['combined'][0.25]['mean']:>9.1f}  "
              f"best={bestk} Δ={diff:+.2f}pp p={p:.3f}{marker}")


if __name__ == "__main__":
    main()
