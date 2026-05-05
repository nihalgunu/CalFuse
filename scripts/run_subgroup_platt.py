"""Multi-seed evaluation of SubgroupStratifiedPlatt vs reference fusions.

Compares:
- Linear-Learned (raw)
- Linear-Learned + Subgroup-Stratified Platt (this paper)
- Multi-CalFuse (HKRR additive corrections)
- CalFuse-Parametric (logit-space)
on the cached BEIR signal matrices, reporting ECE-15 / NDCG@10 /
worst-subgroup ECE-15 with 5-seed CIs.

The point of the comparison: does Multi-CalFuse's additive-logit-
per-cell correction beat a simpler per-subgroup Platt? If they tie,
multi-cal is over-engineered.
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

from src.evaluate import evaluate  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402
from src.fusion.linear_learned import LinearLearnedFusion  # noqa: E402
from src.fusion.multicalibration import (  # noqa: E402
    Multicalibration,
    signal_dominance_subgroups,
    worst_subgroup_ece,
)
from src.fusion.calfuse_conformal import ConformalCalFuse  # noqa: E402
from src.fusion.subgroup_platt import (  # noqa: E402
    SubgroupStratifiedPlatt,
    SubgroupStratifiedIsotonic,
)
from scripts.eval_from_npz import query_level_split  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def make_methods():
    return {
        "linear_learned": lambda: LinearLearnedFusion(),
        "linear_learned_subgroup_platt": lambda: SubgroupStratifiedPlatt(
            base=LinearLearnedFusion(), subgroup_fn=signal_dominance_subgroups()),
        "linear_learned_subgroup_isotonic": lambda: SubgroupStratifiedIsotonic(
            base=LinearLearnedFusion(), subgroup_fn=signal_dominance_subgroups()),
        "calfuse_parametric": lambda: CalFuseFusion(force_mode="parametric"),
        "calfuse_multical": lambda: Multicalibration(
            base=CalFuseFusion(force_mode="parametric"),
            subgroup_fn=signal_dominance_subgroups()),
        "calfuse_conformal": lambda: ConformalCalFuse(
            base=CalFuseFusion(force_mode="parametric"),
            subgroup_fn=signal_dominance_subgroups()),
        "calfuse_parametric_subgroup_platt": lambda: SubgroupStratifiedPlatt(
            base=CalFuseFusion(force_mode="parametric"),
            subgroup_fn=signal_dominance_subgroups()),
        "calfuse_parametric_subgroup_isotonic": lambda: SubgroupStratifiedIsotonic(
            base=CalFuseFusion(force_mode="parametric"),
            subgroup_fn=signal_dominance_subgroups()),
    }


def evaluate_one(npz_path, seed):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    graded = data["graded"].astype(np.float64)
    qids = list(data["qids"])
    split = query_level_split(qids, seed)
    cal = split == "calibration"
    test = split == "test"
    qids_cal = [qids[i] for i in range(len(qids)) if cal[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test[i]]

    M = np.asarray(signal_dominance_subgroups()(X[test], qids_test), dtype=bool)
    out = {}
    for name, factory in make_methods().items():
        try:
            f = factory()
            f.fit(X[cal], y[cal], query_ids=qids_cal)
            p = f.fuse(X[test], query_ids=qids_test)
            ev = evaluate(p, y[test], graded_labels=graded[test],
                          query_ids=qids_test, include_reliability=False).as_dict()
            ev["worst_subgroup_ece_15"] = worst_subgroup_ece(p, y[test], M, n_bins=15, n_min=25)
            out[name] = ev
        except Exception as e:
            out[name] = {"error": repr(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+",
                    default=["nfcorpus", "scifact", "fiqa", "trec-covid",
                             "arguana", "scidocs", "touche-2020"])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--out", default="eval/subgroup_platt_ablation.json")
    args = ap.parse_args()

    aggregate = {}
    for subset in args.subsets:
        npz = Path(f"eval/beir_{subset}_results.npz")
        if not npz.exists():
            print(f"skip {subset}: no npz")
            continue
        per_seed = {}
        for seed in args.seeds:
            per_seed[seed] = evaluate_one(npz, seed)
        # Aggregate.
        method_names = sorted({m for s in per_seed.values() for m in s})
        agg = {}
        for m in method_names:
            for met in ["ece_15", "ndcg_10", "worst_subgroup_ece_15", "brier", "nll"]:
                vals = [per_seed[s][m].get(met) for s in args.seeds
                        if isinstance(per_seed[s][m].get(met), (int, float))]
                if not vals:
                    continue
                arr = np.array(vals, dtype=np.float64)
                agg.setdefault(m, {})[met] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "n": len(arr),
                }
        aggregate[subset] = {"per_seed": per_seed, "aggregated": agg}

        print(f"\n=== {subset} ===")
        print(f"{'method':<40} {'ECE15':>14} {'NDCG10':>14} {'worstECE':>14}")
        for m in method_names:
            a = agg.get(m, {})
            e = a.get("ece_15", {}); n = a.get("ndcg_10", {})
            w = a.get("worst_subgroup_ece_15", {})
            if not e:
                continue
            print(f"{m:<40} {e['mean']:.4f}±{e['std']:.4f}  "
                  f"{n['mean']:.4f}±{n['std']:.4f}  "
                  f"{w['mean']:.4f}±{w['std']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
