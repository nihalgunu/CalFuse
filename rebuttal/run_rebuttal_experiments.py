"""Rebuttal-phase experiments, all computed from frozen artifacts in eval/.

No GPU, no BEIR downloads, no LLM calls. Outputs land in rebuttal/evidence/.

E1  Paired t-tests (by seed) for hallucination-among-non-refused at full
    coverage, all five subsets with verdicts, CalFuse-P / CalFuse vs
    Linear-Learned and vs strongest single retriever; plus paired tests on
    mean-rank-of-first-positive (mechanism check).
E2  Threshold-transfer: at fixed tau, per-dominance-stratum gap between
    realized precision P(y=1 | p>=tau, g) and predicted mean probability,
    multi-seed over 5 reseeded splits x 7 subsets.
E3  Selective-NDCG@10 on nfcorpus and trec-covid with query-bootstrap 95% CIs
    (curve per method + AUC difference CalFuse vs Linear-Learned).
E4  Stratum audit: per-subset dominance-stratum sizes / positive counts on the
    calibration split and whether the pooled-IVAP fallback triggers.
E5  Scorecard from eval/family_significance.json: CalFuse vs Subgroup-Platt
    and vs Subgroup-Isotonic per subset (win / tie / loss at p<0.05).

Usage:  PYTHONPATH=. python3 rebuttal/run_rebuttal_experiments.py
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

from scripts.eval_from_npz import build_methods, query_level_split  # noqa: E402
from src.fusion.multicalibration import signal_dominance_subgroups  # noqa: E402

OUT = REPO / "rebuttal" / "evidence"
OUT.mkdir(exist_ok=True)

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]
SIGNAL_COLS = {n: i for i, n in enumerate(SIGNAL_ORDER)}
SEEDS = [2026, 2027, 2028, 2029, 2030]
HALLU_SUBSETS = ["nfcorpus", "scifact", "fiqa", "arguana", "scidocs"]
ALL_SUBSETS = ["nfcorpus", "scifact", "fiqa", "arguana", "scidocs", "trec-covid", "touche-2020"]
E2_METHODS = ["bm25_platt", "bge_platt", "rrf", "linear_learned",
              "calfuse_parametric", "calfuse_conformal"]


def paired_t(a, b):
    """Paired t-test a vs b (per-seed vectors). Returns dict."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(d) < 2 or np.allclose(d.std(ddof=1), 0):
        return dict(mean_diff=float(d.mean()), t=float("nan"), p=float("nan"), d=float("nan"), n=len(d))
    t, p = sst.ttest_rel(a, b)
    return dict(mean_diff=float(d.mean()), t=float(t), p=float(p),
                d=float(d.mean() / d.std(ddof=1)), n=len(d))


# ---------------------------------------------------------------- E1
def e1_hallucination_tests():
    res = {}
    for subset in HALLU_SUBSETS:
        per_method = {}
        for seed in SEEDS:
            f = REPO / f"eval/multiseed/llm_hallu_{subset}_seed{seed}.json"
            d = json.load(open(f))
            for m, r in d["methods"].items():
                answered = r["grounded"] + r["on_retrieval_only"] + r["fabricated"]
                h = 100.0 * (r["on_retrieval_only"] + r["fabricated"]) / answered if answered else np.nan
                per_method.setdefault(m, []).append(h)
        entry = {m: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), per_seed=v)
                 for m, v in per_method.items()}
        singles = {m: entry[m]["mean"] for m in ("bm25_platt", "bge_platt") if m in entry}
        best_single = min(singles, key=singles.get)
        tests = {}
        for cf in ("calfuse_parametric", "calfuse_conformal"):
            if cf not in per_method:
                continue
            tests[f"{cf}__vs__linear_learned"] = paired_t(per_method[cf], per_method["linear_learned"])
            tests[f"{cf}__vs__{best_single}"] = paired_t(per_method[cf], per_method[best_single])
        res[subset] = dict(hallu_nr_full_coverage=entry, paired_tests=tests,
                           best_single=best_single)

    # Mechanism check from cached mean-rank file: rank-of-first-positive.
    ranks = json.load(open(REPO / "eval/mean_rank_positive_multiseed.json"))
    mech = {}
    for subset, methods in ranks.items():
        try:
            a = methods["calfuse_parametric"]["mean_rank"]
            b = methods["linear_learned"]["mean_rank"]
            # File stores aggregates; per-seed raws only if present.
            if "raw" in a:
                mech[subset] = paired_t(a["raw"], b["raw"])
            else:
                mech[subset] = dict(calfuse_mean=a["mean"], calfuse_sd=a["std"],
                                    linear_mean=b["mean"], linear_sd=b["std"])
        except KeyError:
            continue
    res["_mechanism_mean_rank"] = mech
    json.dump(res, open(OUT / "e1_hallu_paired_tests.json", "w"), indent=1)
    return res


# ---------------------------------------------------------------- E2
def dominance_groups(X):
    M = signal_dominance_subgroups()(X, [])
    return M  # (n, 3) boolean


def e2_threshold_transfer(taus=(0.5, 0.7), min_bucket=20):
    res = {}
    for subset in ALL_SUBSETS:
        npz = np.load(REPO / f"eval/beir_{subset}_results.npz", allow_pickle=True)
        X = npz["X"].astype(np.float64)
        y = npz["y"].astype(np.int64)
        qids = list(npz["qids"])
        sub_res = {m: {str(t): {"worst_gap": [], "spread": [], "marginal_gap": []}
                       for t in taus} for m in E2_METHODS}
        for seed in SEEDS:
            split = np.asarray(query_level_split(qids, seed))
            cal, test = split == "calibration", split == "test"
            qc = [q for q, s in zip(qids, split) if s == "calibration"]
            qt = [q for q, s in zip(qids, split) if s == "test"]
            methods = build_methods(SIGNAL_COLS)
            Mtest = dominance_groups(X[test])
            for m in E2_METHODS:
                f = methods[m]
                f.fit(X[cal], y[cal], query_ids=qc)
                p = np.asarray(f.fuse(X[test], query_ids=qt), float)
                yt = y[test]
                for t in taus:
                    above = p >= t
                    gaps, precs = [], []
                    for g in range(Mtest.shape[1]):
                        sel = above & Mtest[:, g]
                        if sel.sum() < min_bucket:
                            continue
                        realized = float(yt[sel].mean())
                        predicted = float(p[sel].mean())
                        gaps.append(abs(realized - predicted))
                        precs.append(realized)
                    if above.sum() >= min_bucket:
                        marg = abs(float(yt[above].mean()) - float(p[above].mean()))
                    else:
                        marg = np.nan
                    st = sub_res[m][str(t)]
                    if gaps:
                        st["worst_gap"].append(max(gaps))
                        st["spread"].append(max(precs) - min(precs) if len(precs) > 1 else 0.0)
                    if not np.isnan(marg):
                        st["marginal_gap"].append(marg)
        # Aggregate to mean/sd, keep per-seed values for paired tests.
        agg = {}
        for m, by_tau in sub_res.items():
            agg[m] = {}
            for t, st in by_tau.items():
                agg[m][t] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                     n_seeds=len(v), per_seed=[float(x) for x in v]) if v else None
                             for k, v in st.items()}
        # Paired test on worst-stratum gap: CalFuse vs Linear-Learned (only
        # seeds where both produced a measurable stratum).
        tests = {}
        for t in taus:
            a = sub_res["calfuse_conformal"][str(t)]["worst_gap"]
            b = sub_res["linear_learned"][str(t)]["worst_gap"]
            n = min(len(a), len(b))
            if n >= 3:
                tests[str(t)] = paired_t(a[:n], b[:n])
        agg["_paired_worst_gap_calfuse_vs_linear"] = tests
        res[subset] = agg
        print(f"E2 {subset}: done")
    json.dump(res, open(OUT / "e2_threshold_transfer.json", "w"), indent=1)
    return res


# ---------------------------------------------------------------- E3
def e3_selective_ndcg_ci(subsets=("nfcorpus", "trec-covid"), n_boot=2000,
                         coverages=(0.10, 0.25, 0.50, 0.75, 1.00)):
    from scripts.selective_ndcg import per_query_ndcg_and_confidence
    rng = np.random.default_rng(2026)
    res = {}
    focus = ["bge_platt", "linear_learned", "calfuse_parametric", "calfuse_conformal"]
    for subset in subsets:
        npz = np.load(REPO / f"eval/beir_{subset}_results.npz", allow_pickle=True)
        X = npz["X"].astype(np.float64)
        y = npz["y"].astype(np.int64)
        graded = npz["graded"].astype(np.float64)
        qids = list(npz["qids"])
        split = np.asarray(npz["split"])
        cal, test = split == "calibration", split == "test"
        qc = [q for q, s in zip(qids, split) if s == "calibration"]
        qt = [q for q, s in zip(qids, split) if s == "test"]
        methods = build_methods(SIGNAL_COLS)
        per_method_pq = {}
        for m in focus:
            f = methods[m]
            f.fit(X[cal], y[cal], query_ids=qc)
            p = f.fuse(X[test], query_ids=qt)
            per_method_pq[m] = per_query_ndcg_and_confidence(p, graded[test], qt, k=10)

        def curve_from(pq, idx_qs):
            items = sorted(((pq[q]["max_prob"], pq[q]["ndcg"]) for q in idx_qs
                            if not np.isnan(pq[q]["ndcg"])), key=lambda t: -t[0])
            n = len(items)
            return {c: float(np.mean([v for _, v in items[:max(1, int(round(c * n)))]]))
                    for c in coverages}

        qs = sorted(set(qt))
        out = {"n_test_queries": len(qs), "methods": {}}
        for m, pq in per_method_pq.items():
            base = curve_from(pq, qs)
            boots = {c: [] for c in coverages}
            for _ in range(n_boot):
                sample = list(rng.choice(qs, size=len(qs), replace=True))
                cv = curve_from(pq, sample)
                for c in coverages:
                    boots[c].append(cv[c])
            out["methods"][m] = {
                str(c): dict(ndcg=base[c],
                             ci_lo=float(np.percentile(boots[c], 2.5)),
                             ci_hi=float(np.percentile(boots[c], 97.5)))
                for c in coverages}
        # Paired AUC difference CalFuse vs Linear-Learned over bootstrap.
        diffs = []
        for _ in range(n_boot):
            sample = list(rng.choice(qs, size=len(qs), replace=True))
            ca = curve_from(per_method_pq["calfuse_conformal"], sample)
            cb = curve_from(per_method_pq["linear_learned"], sample)
            xs = sorted(coverages)
            diffs.append(float(np.trapezoid([ca[c] - cb[c] for c in xs], xs)))
        diffs = np.asarray(diffs)
        out["auc_diff_calfuse_minus_linear"] = dict(
            mean=float(diffs.mean()),
            ci_lo=float(np.percentile(diffs, 2.5)),
            ci_hi=float(np.percentile(diffs, 97.5)),
            frac_positive=float((diffs > 0).mean()))
        res[subset] = out
        print(f"E3 {subset}: done")
    json.dump(res, open(OUT / "e3_selective_ndcg_ci.json", "w"), indent=1)
    return res


# ---------------------------------------------------------------- E4
def e4_stratum_audit(min_stratum=30, min_positives=5):
    res = {}
    for subset in ALL_SUBSETS:
        npz = np.load(REPO / f"eval/beir_{subset}_results.npz", allow_pickle=True)
        X = npz["X"].astype(np.float64)
        y = npz["y"].astype(np.int64)
        split = np.asarray(npz["split"])
        cal = split == "calibration"
        M = dominance_groups(X[cal])
        yc = y[cal]
        strata = {}
        for g in range(M.shape[1]):
            n = int(M[:, g].sum())
            npos = int(yc[M[:, g]].sum())
            strata[SIGNAL_ORDER[g]] = dict(
                n_cal_pairs=n, n_positives=npos,
                fallback_to_pooled=bool(n < min_stratum or npos < min_positives))
        unassigned = int((~M.any(axis=1)).sum())
        res[subset] = dict(strata=strata, n_cal_pairs=int(cal.sum()),
                           unassigned_pairs=unassigned)
    json.dump(res, open(OUT / "e4_stratum_audit.json", "w"), indent=1)
    return res


# ---------------------------------------------------------------- E5
def e5_scorecard(alpha=0.05):
    fam = json.load(open(REPO / "eval/family_significance.json"))
    ws = fam["worst_subgroup_ece_15"]
    card = {}
    for subset, comps in ws.items():
        row = {}
        for key, r in comps.items():
            a, b = key.split("__vs__")
            if "calfuse_conformal" not in (a, b):
                continue
            # Orient so diff = conformal - other  (negative = conformal better).
            diff = r["mean_diff"] if a == "calfuse_conformal" else -r["mean_diff"]
            other = b if a == "calfuse_conformal" else a
            verdict = "tie"
            if r["p"] < alpha:
                verdict = "win" if diff < 0 else "loss"
            row[other] = dict(diff_conformal_minus_other=float(diff),
                              p=r["p"], verdict=verdict)
        card[subset] = row
    # Tally.
    tally = {}
    for subset, row in card.items():
        for other, r in row.items():
            tally.setdefault(other, {"win": 0, "tie": 0, "loss": 0})[r["verdict"]] += 1
    out = dict(per_subset=card, tally=tally)
    json.dump(out, open(OUT / "e5_scorecard.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    print("== E1: hallucination paired tests ==")
    e1 = e1_hallucination_tests()
    for s in HALLU_SUBSETS:
        for k, t in e1[s]["paired_tests"].items():
            print(f"  {s:10s} {k:45s} diff={t['mean_diff']:+6.2f}pp p={t['p']:.3f} d={t['d']:+.2f}")
    print("\n== E5: scorecard ==")
    e5 = e5_scorecard()
    print(json.dumps(e5["tally"], indent=1))
    print("\n== E4: stratum audit ==")
    e4 = e4_stratum_audit()
    for s, r in e4.items():
        fallbacks = [g for g, v in r["strata"].items() if v["fallback_to_pooled"]]
        print(f"  {s:12s} strata={ {g: v['n_cal_pairs'] for g, v in r['strata'].items()} } fallback={fallbacks or 'none'}")
    print("\n== E3: selective NDCG with CIs ==")
    e3 = e3_selective_ndcg_ci()
    for s, r in e3.items():
        d = r["auc_diff_calfuse_minus_linear"]
        print(f"  {s}: AUC diff CalFuse-LL = {d['mean']:+.4f} [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}] frac+={d['frac_positive']:.2f}")
    print("\n== E2: threshold transfer (slow) ==")
    e2 = e2_threshold_transfer()
    for s in ALL_SUBSETS:
        for m in ("linear_learned", "calfuse_conformal"):
            g = e2[s][m]["0.7"]["worst_gap"]
            if g:
                print(f"  {s:12s} {m:20s} worst-stratum gap@0.7 = {g['mean']:.3f} ± {g['sd']:.3f}")
    print("\nAll outputs in rebuttal/evidence/")
