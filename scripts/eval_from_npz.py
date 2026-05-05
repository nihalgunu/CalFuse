"""Re-evaluate fusion methods directly from a cached .npz signal matrix.

The cached `.npz` files (`eval/beir_*_results.npz`) contain the raw signal
matrix `X`, binary labels `y`, graded labels, query ids, and the
calibration/validation/test split. Re-running the full BEIR evaluation
takes hours of dense-encoder + cross-encoder compute; for fusion-only
ablations (calibrator swap, multicalibration variant, conformal fix
re-test, multi-seed re-splits) we can re-run in seconds from the cache.

Usage:
    PYTHONPATH=. python3 scripts/eval_from_npz.py \
        --npz eval/beir_scifact_results.npz \
        --methods calfuse_conformal \
        [--seed 2026] [--reseed-split]

When `--reseed-split` is passed we re-split queries with the given seed
instead of using the cached ``split`` column. This is how multi-seed
runs are produced without paying signal-computation cost.
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

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def build_methods(signal_cols, include_copula=False):
    bm25_col = signal_cols["bm25"]
    bge_col = signal_cols["dense_bge"]
    e5_col = signal_cols["dense_e5"]
    ce_col = signal_cols["cross_encoder"]
    methods = {
        "bm25_platt": SingleSignalFusion(bm25_col, name="bm25_platt"),
        "bge_platt": SingleSignalFusion(bge_col, name="bge_platt"),
        "e5_platt": SingleSignalFusion(e5_col, name="e5_platt"),
        "cross_encoder_platt": SingleSignalFusion(ce_col, name="cross_encoder_platt"),
        "rrf": RRFFusion(),
        "linear_learned": LinearLearnedFusion(),
        "reranker_fusion": RerankerFusion(reranker_col=ce_col),
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
    if include_copula:
        methods["calfuse_copula"] = CopulaCalFuse(shrinkage=0.2)
    return methods


def query_level_split(qids, seed):
    """Reproduce eval/beir_loader.py's 50/20/30 split with a new seed."""
    unique_qs = sorted(set(qids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_qs))
    n_cal = int(0.5 * len(unique_qs))
    n_val = int(0.2 * len(unique_qs))
    cal_q = set(unique_qs[i] for i in perm[:n_cal])
    val_q = set(unique_qs[i] for i in perm[n_cal:n_cal + n_val])
    out = []
    for q in qids:
        if q in cal_q:
            out.append("calibration")
        elif q in val_q:
            out.append("validation")
        else:
            out.append("test")
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Subset of methods to evaluate (default: all)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reseed-split", action="store_true",
                    help="Re-split queries with --seed instead of using cached split")
    ap.add_argument("--include-copula", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    graded = data["graded"].astype(np.float64)
    qids = list(data["qids"])
    if args.reseed_split:
        split = query_level_split(qids, args.seed)
    else:
        split = np.asarray(data["split"])

    cal_mask = split == "calibration"
    test_mask = split == "test"
    qids_cal = [qids[i] for i in range(len(qids)) if cal_mask[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test_mask[i]]
    X_cal, y_cal = X[cal_mask], y[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    graded_test = graded[test_mask]

    signal_cols = {name: i for i, name in enumerate(SIGNAL_ORDER)}
    methods = build_methods(signal_cols, include_copula=args.include_copula)
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}

    M = np.asarray(signal_dominance_subgroups()(X_test, qids_test), dtype=bool)

    results = []
    for name, fusion in methods.items():
        fusion.fit(X_cal, y_cal, query_ids=qids_cal)
        p = fusion.fuse(X_test, query_ids=qids_test)
        ev = evaluate(p, y_test, graded_labels=graded_test, query_ids=qids_test,
                      include_reliability=False).as_dict()
        ev["method"] = name
        ev["worst_subgroup_ece_15"] = worst_subgroup_ece(p, y_test, M, n_bins=15, n_min=25)
        if hasattr(fusion, "predict_envelope"):
            env = fusion.predict_envelope(X_test, query_ids=qids_test)
            ev["envelope_mean_width"] = float(np.mean(env.p_hi - env.p_lo))
        results.append(ev)
        print(f"{name:24s}  ECE15={ev['ece_15']:.4f}  NDCG@10={ev['ndcg_10'] or 0:.4f}  "
              f"worstECE={ev['worst_subgroup_ece_15']:.4f}"
              + (f"  envW={ev.get('envelope_mean_width', 0):.4f}"
                 if 'envelope_mean_width' in ev else ''))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"npz": args.npz, "seed": args.seed,
                       "reseed_split": args.reseed_split, "methods": results}, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
