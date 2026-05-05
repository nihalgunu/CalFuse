"""Calibration-drift diagnostic.

Tests whether the distribution of per-signal raw scores on the test
split matches the distribution on the calibration-fit split. If it
does not — a phenomenon sometimes called *calibration drift* (Ovadia
et al., 2019, "Can You Trust Your Model's Uncertainty?") — then
calibrators fit on one split will systematically mis-predict on the
other, and the composite estimator will no longer be calibrated even
if Theorem 1's assumptions hold point-wise.

We run a two-sample Kolmogorov-Smirnov test (Massey, 1951) per signal
and report the maximum statistic and minimum p-value across signals.
Holm-Bonferroni handles the multiple-comparison correction. Using KS
rather than the more powerful Anderson-Darling test is a deliberate
choice: KS is distribution-free and returns calibration-friendly
percentiles without any assumption on the tail behaviour of the raw
scores, which varies widely across signal families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationDriftResult:
    per_signal_statistics: np.ndarray
    per_signal_p_values: np.ndarray
    reject: bool
    worst_signal: int


def calibration_drift_test(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float = 0.05,
) -> CalibrationDriftResult:
    from scipy.stats import ks_2samp

    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    n_signals = calibration_scores.shape[1]

    stats = np.zeros(n_signals)
    pvals = np.ones(n_signals)
    for j in range(n_signals):
        res = ks_2samp(calibration_scores[:, j], test_scores[:, j])
        stats[j] = float(res.statistic)
        pvals[j] = float(res.pvalue)

    # Holm-Bonferroni: reject only the signals whose p-value beats the
    # per-signal adjusted threshold.
    order = np.argsort(pvals)
    reject = False
    for rank, j in enumerate(order):
        thresh = alpha / (n_signals - rank)
        if pvals[j] <= thresh:
            reject = True
        else:
            break

    return CalibrationDriftResult(
        per_signal_statistics=stats,
        per_signal_p_values=pvals,
        reject=bool(reject),
        worst_signal=int(np.argmax(stats)),
    )
