"""Run the three diagnostic tests on cached BEIR signal matrices.

Tests:
- Signal-dependence (Fisher-z, Holm-Bonferroni) on calibrated logits
- Calibration-drift (KS, Holm-Bonferroni) between cal/test raw scores
- Fusion-rule mismatch (Wilks chi^2) between parametric and learned

Reports a per-subset table that ties each diagnostic to a concrete
auto-switch decision so the experimental section can claim diagnostics
work on real data, not only on the synthetic substrates.
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

from src.calibrators.platt import PlattCalibrator  # noqa: E402
from src.diagnostics.calibration_drift import calibration_drift_test  # noqa: E402
from src.diagnostics.fusion_mismatch import fusion_mismatch_test  # noqa: E402
from src.diagnostics.signal_dependence import signal_dependence_test  # noqa: E402


SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def run(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    split = np.asarray(data["split"])
    cal = split == "calibration"
    test = split == "test"

    # Per-signal Platt calibration on the calibration split.
    p_cal = np.zeros_like(X[cal])
    for j in range(X.shape[1]):
        c = PlattCalibrator()
        c.fit(X[cal, j], y[cal])
        p_cal[:, j] = c.transform(X[cal, j])

    dep = signal_dependence_test(p_cal, y[cal], alpha=0.05)
    drift = calibration_drift_test(X[cal], X[test], alpha=0.05)
    mism = fusion_mismatch_test(X[cal], y[cal], alpha=0.05)

    out = {
        "signal_dependence": {
            "max_abs_z": float(dep.statistic),
            "min_p_value": float(dep.p_value),
            "reject": bool(dep.reject),
            "max_pair_corr": float(np.max(dep.per_pair_correlations)),
            "off_diagonal_mass": float(dep.off_diagonal_mass),
        },
        "calibration_drift": {
            "per_signal_ks": dep_arr_to_dict(SIGNAL_ORDER, drift.per_signal_statistics),
            "per_signal_p": dep_arr_to_dict(SIGNAL_ORDER, drift.per_signal_p_values),
            "reject": bool(drift.reject),
            "worst_signal": SIGNAL_ORDER[int(drift.worst_signal)],
        },
        "fusion_mismatch": {
            "chi2": float(getattr(mism, "statistic", float("nan"))),
            "p_value": float(getattr(mism, "p_value", float("nan"))),
            "reject": bool(getattr(mism, "reject", False)),
        },
        # The auto-switch policy CalFuse would have used.
        "auto_switch_decision": (
            "copula" if dep.reject and float(np.max(dep.per_pair_correlations)) > 0.3
            else "learned" if dep.reject
            else "parametric"
        ),
    }
    return out


def dep_arr_to_dict(names, arr):
    return {n: float(arr[i]) for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--out", default="eval/diagnostics_beir.json")
    args = ap.parse_args()

    out = {}
    for path in args.npz:
        name = Path(path).stem.replace("beir_", "").replace("_results", "")
        print(f"\n=== {name} ===")
        res = run(Path(path))
        out[name] = res
        sd = res["signal_dependence"]
        cd = res["calibration_drift"]
        fm = res["fusion_mismatch"]
        print(f"  signal-dependence:  max|z|={sd['max_abs_z']:.2f}  "
              f"min_p={sd['min_p_value']:.4g}  reject={sd['reject']}  "
              f"max_corr={sd['max_pair_corr']:.3f}")
        print(f"  cal-drift:          worst={cd['worst_signal']}  reject={cd['reject']}")
        print(f"  fusion-mismatch:    chi2={fm['chi2']:.2f}  p={fm['p_value']:.4g}  "
              f"reject={fm['reject']}")
        print(f"  -> auto-switch:    {res['auto_switch_decision']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
