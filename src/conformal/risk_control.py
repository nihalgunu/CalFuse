"""Conformal risk control for fused retrieval scores (H3).

Framework
---------
Conformal risk control (Angelopoulos, Bates, Fisch, Lei, Schuster
2022, ``Conformal risk control'') extends conformal coverage to
user-specified monotone loss functions. Given an i.i.d./exchangeable
calibration sample, CRC chooses a threshold $\\lambda^\\star$ on a
scoring function such that the *expected loss* on new data is at
most a user-specified $r$, with finite-sample guarantees.

Retrieval application
---------------------
For selective retrieval, the natural loss is 0-1 selective error:
the predictor either returns a decision (include / exclude) or
abstains; if it decides, the loss is 1 iff the decision is wrong.
CRC chooses an abstention threshold on the fused probability such
that the expected selective error on future queries stays below $r$
with probability $1 - \\delta$.

Concretely, given a calibration sample $(X_i, Y_i)_{i=1}^n$ of
query--passage pairs with binary relevance labels, let
$\\ell_\\lambda(X, Y) = \\mathbf{1}[|p(X) - 0.5| \\geq \\lambda]
\\cdot \\mathbf{1}[\\text{sign}(p(X) - 0.5) \\neq Y']$, where $Y'$
is the $\\{-1, +1\\}$-coded label. The CRC choice
$\\hat{\\lambda}(r) = \\inf\\{\\lambda : \\widehat{R}_n(\\lambda) \\leq r - (1-r)/n\\}$
guarantees $\\expec[\\ell_{\\hat{\\lambda}}(X_{n+1}, Y_{n+1})] \\leq r$
under exchangeability (ABFSL 2022, Theorem 1).

Status
------
Stub implementation: CRC threshold fitting + evaluation on a
held-out split. Pass criterion E7-H3 in ``RUN_EXPERIMENTS.md``
compares CRC-certified selective accuracy to the best uncertified
operating point.

References
----------
* Angelopoulos, Bates, Fisch, Lei, Schuster 2022, ``Conformal risk
  control'', ICLR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CRCResult:
    lambda_star: float
    achieved_risk: float
    coverage: float  # fraction of inputs on which the predictor decides
    selective_accuracy: float  # accuracy on covered inputs


def _selective_loss(probs: np.ndarray, labels: np.ndarray, lam: float) -> tuple[float, float, float]:
    """Return (risk, coverage, selective_accuracy) at threshold lam."""
    confidence = np.abs(probs - 0.5)
    decide = confidence >= lam
    if decide.sum() == 0:
        return 0.0, 0.0, 0.0
    preds = (probs >= 0.5).astype(np.int64)
    # Risk is fraction of decided-on pairs that are wrong.
    wrong = (preds != labels.astype(np.int64))
    risk = float((decide & wrong).sum() / decide.sum())
    coverage = float(decide.mean())
    selective_accuracy = 1.0 - risk
    return risk, coverage, selective_accuracy


def conformal_risk_control(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    nominal_risk: float,
    lam_grid: np.ndarray | None = None,
) -> CRCResult:
    """Fit the CRC abstention threshold on calibration data.

    Uses the ABFSL finite-sample correction: we require
    $\\widehat{R}_n(\\lambda) \\leq r - (1-r)/n$ so that the
    expected risk on a fresh exchangeable sample stays below $r$.

    Parameters
    ----------
    cal_probs, cal_labels
        Calibration fused probabilities and binary labels.
    nominal_risk : r
        User-specified expected-selective-error upper bound.
    lam_grid
        Optional search grid over $[0, 0.5]$; default 201 points.
    """
    n = cal_probs.shape[0]
    if lam_grid is None:
        lam_grid = np.linspace(0.0, 0.5, 201)
    target = nominal_risk - (1.0 - nominal_risk) / max(1, n)
    best_lam = 0.5  # degenerate: abstain always -> zero risk, zero coverage
    best_cov = 0.0
    for lam in sorted(lam_grid):
        risk, cov, _ = _selective_loss(cal_probs, cal_labels, lam)
        if risk <= target and cov > best_cov:
            # Prefer the smallest lam (largest coverage) meeting the risk bound.
            best_lam = float(lam)
            best_cov = cov
    risk, cov, sa = _selective_loss(cal_probs, cal_labels, best_lam)
    return CRCResult(
        lambda_star=best_lam,
        achieved_risk=risk,
        coverage=cov,
        selective_accuracy=sa,
    )


def evaluate_crc_at(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    nominal_risk: float,
) -> tuple[CRCResult, CRCResult]:
    """Fit CRC on calibration, evaluate on test. Returns both the
    calibration-split summary and the test-split achieved numbers
    so callers can report the CRC validity gap.
    """
    cal_res = conformal_risk_control(cal_probs, cal_labels, nominal_risk)
    test_risk, test_cov, test_sa = _selective_loss(
        test_probs, test_labels, cal_res.lambda_star
    )
    test_res = CRCResult(
        lambda_star=cal_res.lambda_star,
        achieved_risk=test_risk,
        coverage=test_cov,
        selective_accuracy=test_sa,
    )
    return cal_res, test_res
