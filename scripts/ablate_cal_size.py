"""Calibration-set-size scaling ablation — reads the cached ``.npz`` signal
matrix from a prior ``evaluate_beir`` run and refits the three production
CalFuse variants on progressively smaller subsamples of the calibration
split. Answers the E4 question ``how much calibration data does CalFuse
actually need before its calibration improvement stabilises?``

The subsample is query-level (not pair-level): we fix a random subset of
calibration queries, then use every pair from those queries. This matches
how calibration sets are collected in practice (you annotate queries, not
individual pairs).

Usage
-----
```
PYTHONPATH=. python3 scripts/ablate_cal_size.py \\
    --npz eval/beir_nfcorpus_results.npz \\
    --out eval/ablate_cal_size_nfcorpus.json
```
"""

from __future__ import annotations

import argparse
import json
import random
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

CAL_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def _subsample_cal_mask(cal_mask: np.ndarray, qids: list, fraction: float, seed: int):
    cal_qids_unique = sorted({qids[i] for i in range(len(qids)) if cal_mask[i]})
    rng = random.Random(seed)
    rng.shuffle(cal_qids_unique)
    n_keep = max(1, int(round(fraction * len(cal_qids_unique))))
    kept = set(cal_qids_unique[:n_keep])
    return np.array([(cal_mask[i] and qids[i] in kept) for i in range(len(qids))],
                    dtype=bool), n_keep


def main():
    parser = argparse.ArgumentParser(description="CalFuse cal-size scaling ablation")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out) if args.out else npz_path.with_name(
        f"ablate_cal_size_{npz_path.stem.replace('beir_', '').replace('_results', '')}.json"
    )

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(np.int64)
    graded = data["graded"]
    qids = list(data["qids"])
    split = data["split"]

    cal_mask_full = split == "calibration"
    test_mask = split == "test"
    qids_test = [qids[i] for i in range(len(qids)) if test_mask[i]]

    X_test, y_test = X[test_mask], y[test_mask]
    graded_test = graded[test_mask]

    methods = {
        "linear_learned": lambda: LinearLearnedFusion(),
        "calfuse_parametric": lambda: CalFuseFusion(force_mode="parametric"),
        "calfuse_multical": lambda: Multicalibration(
            base=CalFuseFusion(force_mode="parametric"),
            subgroup_fn=signal_dominance_subgroups()),
    }
    M_test = np.asarray(signal_dominance_subgroups()(X_test, qids_test), dtype=bool)

    results = []
    for frac in CAL_FRACTIONS:
        cal_mask_sub, n_queries_kept = _subsample_cal_mask(
            cal_mask_full, qids, frac, args.seed)
        X_cal, y_cal = X[cal_mask_sub], y[cal_mask_sub]
        qids_cal = [qids[i] for i in range(len(qids)) if cal_mask_sub[i]]

        for name, ctor in methods.items():
            try:
                fusion = ctor()
                fusion.fit(X_cal, y_cal, query_ids=qids_cal)
                p = fusion.fuse(X_test, query_ids=qids_test)
                ev = evaluate(p, y_test, graded_labels=graded_test,
                              query_ids=qids_test, include_reliability=False).as_dict()
                wse = worst_subgroup_ece(p, y_test, M_test, n_bins=15, n_min=25)
                results.append({
                    "cal_fraction": frac,
                    "n_cal_queries": int(n_queries_kept),
                    "n_cal_pairs": int(cal_mask_sub.sum()),
                    "method": name,
                    "ece_15": ev["ece_15"],
                    "ndcg_10": ev.get("ndcg_10"),
                    "worst_subgroup_ece_15": wse,
                })
                print(f"  frac={frac:.2f} (n_q={n_queries_kept:4d})  "
                      f"{name:20s}  ECE={ev['ece_15']:.4f}  "
                      f"NDCG@10={ev.get('ndcg_10', 0):.4f}  worst-sg={wse:.4f}")
            except Exception as e:  # noqa: BLE001
                results.append({"cal_fraction": frac, "method": name, "error": repr(e)})

    out = {
        "_type": "calfuse_cal_size_ablation",
        "source_npz": str(npz_path),
        "n_pairs_test": int(test_mask.sum()),
        "test_positive_rate": float(y_test.mean()),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
