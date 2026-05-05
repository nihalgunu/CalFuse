"""Compute the auxiliary data needed for paper Figures 5 and 6:

Fig 5 (reliability): per-bin (mean predicted prob, observed freq,
count) on nfcorpus for (Linear-Learned, marginal), (Linear-Learned,
worst-subgroup stratum), (CalFuse-Parametric, worst-subgroup stratum).
Subgroups = query-length buckets (short/medium/long). Worst subgroup =
bucket with highest Linear-Learned ECE-15.

Fig 6 (mechanism): per-query rank-of-first-positive in top-5 for each
of the 5 methods on scifact, across 5 seeds. We need the empirical
distribution (not just the mean) to overlay histograms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.beir_loader import build_candidate_pool, load_beir, query_level_splits  # noqa: E402
from scripts.eval_from_npz import build_methods  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]
N_BINS = 15


def equal_mass_bins(probs, n_bins=N_BINS):
    """Return bin edges so each bin holds ~equal mass."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(probs, qs))
    if len(edges) < 2:
        edges = np.array([probs.min(), probs.max()])
    edges[0] = max(0.0, edges[0] - 1e-9)
    edges[-1] = min(1.0, edges[-1] + 1e-9)
    return edges


def reliability_bins(probs, y, edges):
    """Return list of dicts (pred_mean, obs_freq, count)."""
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi if i < len(edges) - 2 else probs <= hi)
        c = int(m.sum())
        if c == 0:
            continue
        out.append(dict(pred=float(probs[m].mean()),
                        obs=float(y[m].mean()),
                        count=c, lo=float(lo), hi=float(hi)))
    return out


def ece(probs, y, edges):
    bins = reliability_bins(probs, y, edges)
    n = sum(b["count"] for b in bins)
    if n == 0: return float("nan")
    return sum(b["count"] * abs(b["pred"] - b["obs"]) for b in bins) / n


def signal_dominance_buckets(X):
    """Match src.fusion.multicalibration.signal_dominance_subgroups().
    Returns per-row bucket id = argmax(standardised signal). Top-3
    signals only (per the multicalibration default)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0); sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    dom = np.argmax(Xs, axis=1)
    # Cap at top-3 like multicalibration default.
    return np.clip(dom, 0, 2)


def main_reliability():
    """Fig 5 data: nfcorpus reliability bins."""
    print("=== nfcorpus reliability ===")
    sub = load_beir("nfcorpus")
    pairs, _, qids_built, _ = build_candidate_pool(
        sub, top_k_bm25=100, max_negatives_per_query=40, seed=2026)
    npz = np.load(REPO / "eval/beir_nfcorpus_results.npz", allow_pickle=True)
    qids = list(npz["qids"]); assert qids == qids_built
    X = npz["X"].astype(np.float64); y = npz["y"].astype(np.int64)
    splits = query_level_splits(qids, seed=2026)
    sa = np.array([splits[q] for q in qids])
    test, cal = sa == "test", sa == "calibration"
    qids_cal = [qids[i] for i in range(len(qids)) if cal[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test[i]]
    qtext = {p.query_id: p.query_text for p in pairs}

    cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    methods = build_methods(cols)

    # Fit Linear-Learned and CalFuse-multical (the paper's "CalFuse").
    out = {}
    for name in ["linear_learned", "calfuse_multical"]:
        f = methods[name]
        f.fit(X[cal], y[cal], query_ids=qids_cal)
        probs = f.fuse(X[test], query_ids=qids_test)
        out[name] = probs
    y_test = y[test]

    # Subgroup family = signal-dominance over top-3 signals
    # (matching multicalibration.py).
    buckets = signal_dominance_buckets(X[test])
    bucket_ece = {}
    for b in [0, 1, 2]:
        m = buckets == b
        if m.sum() < 30: continue
        edges_local = equal_mass_bins(out["linear_learned"][m])
        bucket_ece[b] = ece(out["linear_learned"][m], y_test[m], edges_local)
    worst_b = max(bucket_ece, key=bucket_ece.get)
    print(f"  per-bucket Linear-Learned ECE-15: {bucket_ece}")
    print(f"  worst bucket: {worst_b} (ECE-15={bucket_ece[worst_b]:.4f})")

    bucket_names = {0: "BM25-dominant", 1: "BGE-dominant",
                    2: "E5-dominant"}
    panels = {}

    # (A) Linear-Learned, marginal — all test rows.
    edges = equal_mass_bins(out["linear_learned"])
    panels["A_LL_marginal"] = dict(
        bins=reliability_bins(out["linear_learned"], y_test, edges),
        ece=ece(out["linear_learned"], y_test, edges),
        n=int(test.sum()),
        title="Linear-Learned, marginal",
        subtitle="(looks well-calibrated)",
    )
    # (B) Linear-Learned, worst-bucket only.
    m = buckets == worst_b
    edges_b = equal_mass_bins(out["linear_learned"][m])
    panels["B_LL_worstsg"] = dict(
        bins=reliability_bins(out["linear_learned"][m], y_test[m], edges_b),
        ece=ece(out["linear_learned"][m], y_test[m], edges_b),
        n=int(m.sum()),
        title=f"Linear-Learned, worst stratum: {bucket_names[worst_b]}",
        subtitle="(subgroup miscalibration)",
    )
    # (C) CalFuse-multical, same worst-bucket.
    edges_c = equal_mass_bins(out["calfuse_multical"][m])
    panels["C_CF_worstsg"] = dict(
        bins=reliability_bins(out["calfuse_multical"][m], y_test[m], edges_c),
        ece=ece(out["calfuse_multical"][m], y_test[m], edges_c),
        n=int(m.sum()),
        title=f"CalFuse, worst stratum: {bucket_names[worst_b]}",
        subtitle="(after CalFuse correction)",
    )
    for k, v in panels.items():
        print(f"  {k}: ECE={v['ece']:.4f}, n={v['n']}, {len(v['bins'])} bins")

    Path(REPO / "eval/figdata_reliability_nfcorpus.json").write_text(
        json.dumps(panels, indent=1))
    print("Wrote eval/figdata_reliability_nfcorpus.json")


def main_ranks():
    """Fig 6 data: per-query rank-of-first-positive on scifact, all
    methods, all seeds. We sample per seed = 50 queries × 5 methods.
    The 'rank' is fully determined by the method (not the seed); only
    the query subsample varies. So we compute rank per query for each
    method once, and then count occurrences of each query across the
    seed-50-samples."""
    print("\n=== scifact per-query rank-of-first-positive ===")
    sub = load_beir("scifact")
    pairs, _, qids_built, _ = build_candidate_pool(
        sub, top_k_bm25=100, max_negatives_per_query=40, seed=2026)
    npz = np.load(REPO / "eval/beir_scifact_results.npz", allow_pickle=True)
    qids = list(npz["qids"]); assert qids == qids_built
    X = npz["X"].astype(np.float64); y = npz["y"].astype(np.int64)
    splits = query_level_splits(qids, seed=2026)
    sa = np.array([splits[q] for q in qids])
    test, cal = sa == "test", sa == "calibration"
    qids_cal = [qids[i] for i in range(len(qids)) if cal[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test[i]]
    y_test = y[test]
    by_q = {}
    for i, q in enumerate(qids_test):
        by_q.setdefault(q, []).append(i)

    cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    methods = build_methods(cols)
    out = {}
    for name in ["bm25_platt", "bge_platt", "linear_learned",
                 "calfuse_parametric", "calfuse_conformal"]:
        f = methods[name]
        f.fit(X[cal], y[cal], query_ids=qids_cal)
        probs = f.fuse(X[test], query_ids=qids_test)
        rank_per_q = {}
        for q, idxs in by_q.items():
            idxs = np.asarray(idxs)
            p = probs[idxs]
            order = np.argsort(-p, kind="mergesort")[:5]
            is_pos = [int(y_test[idxs[j]] == 1) for j in order]
            rank_per_q[q] = (is_pos.index(1) + 1) if any(is_pos) else 6
        out[name] = rank_per_q
        ranks = list(rank_per_q.values())
        print(f"  {name}: {len(ranks)} queries, mean rank={np.mean(ranks):.3f}")

    # Aggregate the seed=2026 50-query sample (representative; we
    # showed seed-to-seed std is small in mean_rank_positive_multiseed).
    # Read which queries each seed sampled so we can stack them.
    all_by_method = {m: [] for m in out}
    for seed in [2026, 2027, 2028, 2029, 2030]:
        vfile = REPO / f"eval/multiseed/llm_hallu_scifact_seed{seed}.json"
        d = json.load(open(vfile))
        sampled_qids = [ex["qid"] for ex in d["methods"]["calfuse_parametric"]["examples"]]
        for m in out:
            all_by_method[m].extend([out[m][q] for q in sampled_qids if q in out[m]])
    summary = {m: dict(ranks=v, mean=float(np.mean(v)),
                        std=float(np.std(v, ddof=1)),
                        n=len(v))
                for m, v in all_by_method.items()}
    for m, s in summary.items():
        print(f"  pooled 5-seed: {m:<22} n={s['n']}  mean={s['mean']:.3f}±{s['std']:.3f}")
    Path(REPO / "eval/figdata_ranks_scifact.json").write_text(
        json.dumps(summary, indent=1))
    print("Wrote eval/figdata_ranks_scifact.json")


def main():
    main_reliability()
    main_ranks()


if __name__ == "__main__":
    main()
