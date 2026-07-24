"""Seed-scaling check: rerun the paper's core CPU-side claims at 20 seeds.

The paper's multi-seed protocol uses 5 reseeded query-level splits. Reviewers
call the downstream stats thin; every CPU-side analysis can be scaled to 20
seeds directly from the frozen npz matrices. This reruns:

L1  Worst-subgroup ECE-15 (paper's headline metric) for the CalFuse family,
    the stratified-calibrator baselines, and the reference fusions, on all
    seven subsets, seeds 2026..2045, with paired tests for every comparison
    quoted in the rebuttal.
L2  The threshold-transfer experiment (rebuttal E2) at the same 20 seeds.

Usage:  PYTHONPATH=. python3 rebuttal/run_seed_scaling.py
Writes rebuttal/evidence/l1_seed_scaling_wsece.json and
       rebuttal/evidence/l2_threshold_transfer_20seed.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_from_npz import query_level_split  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402
from src.fusion.calfuse_conformal import ConformalCalFuse  # noqa: E402
from src.fusion.linear_learned import LinearLearnedFusion  # noqa: E402
from src.fusion.multicalibration import (  # noqa: E402
    Multicalibration,
    signal_dominance_subgroups,
    worst_subgroup_ece,
)
from src.fusion.rrf import RRFFusion  # noqa: E402
from src.fusion.subgroup_platt import (  # noqa: E402
    SubgroupStratifiedIsotonic,
    SubgroupStratifiedPlatt,
)
from rebuttal.run_rebuttal_experiments import paired_t  # noqa: E402

OUT = REPO / "rebuttal" / "evidence"
SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]
SEEDS = list(range(2026, 2046))  # 20 seeds, superset of the paper's 5
SUBSETS = ["nfcorpus", "scifact", "fiqa", "arguana", "scidocs", "trec-covid", "touche-2020"]


def make_methods():
    dom = signal_dominance_subgroups
    return {
        "rrf": lambda: RRFFusion(),
        "linear_learned": lambda: LinearLearnedFusion(),
        "calfuse_parametric": lambda: CalFuseFusion(force_mode="parametric"),
        "calfuse_multical": lambda: Multicalibration(
            base=CalFuseFusion(force_mode="parametric"), subgroup_fn=dom()),
        "calfuse_conformal": lambda: ConformalCalFuse(
            base=CalFuseFusion(force_mode="parametric"), subgroup_fn=dom()),
        "calfuse_parametric_subgroup_platt": lambda: SubgroupStratifiedPlatt(
            base=CalFuseFusion(force_mode="parametric"), subgroup_fn=dom()),
        "calfuse_parametric_subgroup_isotonic": lambda: SubgroupStratifiedIsotonic(
            base=CalFuseFusion(force_mode="parametric"), subgroup_fn=dom()),
    }


def l1_wsece():
    res = {}
    for subset in SUBSETS:
        npz = np.load(REPO / f"eval/beir_{subset}_results.npz", allow_pickle=True)
        X = npz["X"].astype(np.float64)
        y = npz["y"].astype(np.int64)
        graded = npz["graded"].astype(np.float64)
        qids = list(npz["qids"])
        per_method = {m: {"worst_subgroup_ece_15": [], "ece_15": [], "ndcg_10": []}
                      for m in make_methods()}
        for seed in SEEDS:
            split = np.asarray(query_level_split(qids, seed))
            cal, test = split == "calibration", split == "test"
            qc = [q for q, s in zip(qids, split) if s == "calibration"]
            qt = [q for q, s in zip(qids, split) if s == "test"]
            M = np.asarray(signal_dominance_subgroups()(X[test], qt), dtype=bool)
            for name, factory in make_methods().items():
                f = factory()
                f.fit(X[cal], y[cal], query_ids=qc)
                p = f.fuse(X[test], query_ids=qt)
                ev = evaluate(p, y[test], graded_labels=graded[test],
                              query_ids=qt, include_reliability=False).as_dict()
                st = per_method[name]
                st["worst_subgroup_ece_15"].append(
                    worst_subgroup_ece(p, y[test], M, n_bins=15, n_min=25))
                st["ece_15"].append(ev["ece_15"])
                st["ndcg_10"].append(ev["ndcg_10"])
        agg = {m: {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                           per_seed=[float(x) for x in v])
                   for k, v in st.items()}
               for m, st in per_method.items()}
        tests = {}
        base = per_method["calfuse_conformal"]["worst_subgroup_ece_15"]
        for other in per_method:
            if other == "calfuse_conformal":
                continue
            tests[f"calfuse_conformal__vs__{other}"] = paired_t(
                base, per_method[other]["worst_subgroup_ece_15"])
        res[subset] = dict(aggregated=agg, paired_wsece=tests)
        best = min(agg, key=lambda m: agg[m]["worst_subgroup_ece_15"]["mean"])
        print(f"L1 {subset}: best={best} "
              f"({agg[best]['worst_subgroup_ece_15']['mean']:.4f}); "
              f"conformal={agg['calfuse_conformal']['worst_subgroup_ece_15']['mean']:.4f} "
              f"linear={agg['linear_learned']['worst_subgroup_ece_15']['mean']:.4f}")
        sys.stdout.flush()
    json.dump(res, open(OUT / "l1_seed_scaling_wsece.json", "w"), indent=1)
    return res


def l2_threshold_transfer_20():
    import rebuttal.run_rebuttal_experiments as rre
    rre.SEEDS = SEEDS
    res = rre.e2_threshold_transfer()
    json.dump(res, open(OUT / "l2_threshold_transfer_20seed.json", "w"), indent=1)
    # The e2 function writes its default file too; restore it from the 5-seed
    # run is unnecessary — keep both artifacts, 20-seed one is authoritative.
    return res


if __name__ == "__main__":
    print(f"Seeds: {SEEDS[0]}..{SEEDS[-1]} ({len(SEEDS)})")
    l1 = l1_wsece()
    print("\nL2: threshold transfer at 20 seeds")
    l2 = l2_threshold_transfer_20()
    for s, agg in l2.items():
        for t, r in agg.get("_paired_worst_gap_calfuse_vs_linear", {}).items():
            print(f"  {s:12s} tau={t}  diff={r['mean_diff']:+.3f} p={r['p']:.4f} d={r['d']:+.2f} (n={r['n']})")
    print("\nDone.")
