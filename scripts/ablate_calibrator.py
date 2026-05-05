"""Per-signal calibrator ablation — reads the cached ``.npz`` signal matrix
from a prior ``evaluate_beir`` run and refits CalFuse under each of
``platt`` / ``isotonic`` / ``temperature`` / ``learned_mlp``. Because the
signal matrix is cached, every additional calibrator costs seconds rather
than re-running the ~100-min signal computation.

Reports marginal ECE-15, NDCG@10, and worst-subgroup ECE-15 per
calibrator, so a single table answers the "is the choice of per-signal
calibrator second-order?" question that the paper's E2 ablation is
supposed to answer.

Usage
-----
```
PYTHONPATH=. python3 scripts/ablate_calibrator.py \\
    --npz eval/beir_nfcorpus_results.npz \\
    --out eval/ablate_calibrator_nfcorpus.json
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

from src.calibrators.isotonic import IsotonicCalibrator  # noqa: E402
from src.calibrators.learned_calibrator import LearnedCalibrator  # noqa: E402
from src.calibrators.platt import PlattCalibrator  # noqa: E402
from src.calibrators.temperature import TemperatureCalibrator  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402
from src.fusion.multicalibration import (  # noqa: E402
    Multicalibration,
    signal_dominance_subgroups,
    worst_subgroup_ece,
)

CALIBRATORS = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "temperature": TemperatureCalibrator,
    "learned_mlp": LearnedCalibrator,
}


def main():
    parser = argparse.ArgumentParser(description="CalFuse calibrator ablation")
    parser.add_argument("--npz", required=True, help="Path to evaluate_beir's .npz output")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out) if args.out else npz_path.with_name(
        f"ablate_calibrator_{npz_path.stem.replace('beir_', '').replace('_results', '')}.json"
    )

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(np.int64)
    graded = data["graded"]
    qids = list(data["qids"])
    split = data["split"]

    cal_mask = split == "calibration"
    test_mask = split == "test"
    qids_cal = [qids[i] for i in range(len(qids)) if cal_mask[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test_mask[i]]

    X_cal, y_cal = X[cal_mask], y[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    graded_test = graded[test_mask]

    results = []
    for cal_name, cal_cls in CALIBRATORS.items():
        for fusion_name, fusion_ctor in [
            ("calfuse_parametric", lambda: CalFuseFusion(
                force_mode="parametric", calibrator_factory=cal_cls)),
            ("calfuse_multical", lambda: Multicalibration(
                base=CalFuseFusion(force_mode="parametric", calibrator_factory=cal_cls),
                subgroup_fn=signal_dominance_subgroups())),
        ]:
            try:
                fusion = fusion_ctor()
                fusion.fit(X_cal, y_cal, query_ids=qids_cal)
                p = fusion.fuse(X_test, query_ids=qids_test)
                ev = evaluate(p, y_test, graded_labels=graded_test,
                              query_ids=qids_test, include_reliability=False).as_dict()
                M = np.asarray(signal_dominance_subgroups()(X_test, qids_test), dtype=bool)
                wse = worst_subgroup_ece(p, y_test, M, n_bins=15, n_min=25)
                results.append({
                    "calibrator": cal_name,
                    "method": fusion_name,
                    "ece_15": ev["ece_15"],
                    "ndcg_10": ev.get("ndcg_10"),
                    "worst_subgroup_ece_15": wse,
                    "brier": ev.get("brier"),
                })
                print(f"  {cal_name:12s} / {fusion_name:20s}  "
                      f"ECE={ev['ece_15']:.4f}  NDCG@10={ev.get('ndcg_10', 0):.4f}  "
                      f"worst-sg={wse:.4f}")
            except Exception as e:  # noqa: BLE001
                results.append({"calibrator": cal_name, "method": fusion_name,
                                "error": repr(e)})

    out = {
        "_type": "calfuse_calibrator_ablation",
        "source_npz": str(npz_path),
        "n_pairs_calibration": int(cal_mask.sum()),
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
