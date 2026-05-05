"""Fusion-rule mismatch diagnostic.

Tests whether the parametric CalFuse form (affine in per-signal logits)
is an adequate model for the data by comparing it to a saturated
alternative (an MLP with the same input features). The test is the
classical likelihood-ratio test (Wilks, 1938): under the null
``2 * (ll_saturated - ll_parametric) ~ chi^2_df`` where ``df`` is the
effective-parameter difference.

We use ``df = hidden + 1`` as a coarse upper bound on the MLP's extra
capacity (the actual effective df is smaller after L2 regularisation,
so the test is conservative — it under-rejects). An under-rejecting
test for *mismatch* is preferable: we only want to switch away from
the parametric form when we have strong evidence against it.

The diagnostic is used to generate a Phase 3 failure-mode experiment,
not as an in-line trigger inside CalFuse itself, because fusion
mismatch is (by design) handled by CalFuse's learned variant when
dependence dictates the switch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionMismatchResult:
    statistic: float
    p_value: float
    degrees_of_freedom: int
    reject: bool


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fusion_mismatch_test(
    calibrated_logits: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
    random_state: int = 0,
) -> FusionMismatchResult:
    """Wilks chi^2 test of parametric fusion vs a quadratic-feature
    saturated alternative.

    The earlier implementation used an MLP-with-LBFGS as the saturated
    alternative; on BEIR-scale calibration sets (~25k pairs) sklearn's
    MLPClassifier(solver="lbfgs") sometimes failed to converge in
    minutes. The quadratic-feature logistic regression has the same
    role here: under the Gaussian-copula model on calibrated logits
    (Theorem 3), the Bayes-optimal log-odds is exactly affine in
    ``L`` and ``L L^T``, so a quadratic-feature logistic is a
    correctly-specified saturated alternative. The chi^2 df is the
    number of additional parameters (the ``d(d+1)/2`` upper-triangle
    entries of the cross-product feature matrix).
    """
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(calibrated_logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    if np.unique(y).size < 2:
        return FusionMismatchResult(0.0, 1.0, 0, False)

    parametric = LogisticRegression(C=10.0, solver="lbfgs", max_iter=200).fit(X, y)
    ll_param = -_nll(parametric.predict_proba(X)[:, 1], y)

    d = X.shape[1]
    # Upper-triangular cross-products including squares.
    iu, ju = np.triu_indices(d)
    quad = X[:, iu] * X[:, ju]
    Xq = np.concatenate([X, quad], axis=1)
    saturated = LogisticRegression(C=10.0, solver="lbfgs", max_iter=200).fit(Xq, y)
    ll_sat = -_nll(saturated.predict_proba(Xq)[:, 1], y)

    stat = max(0.0, 2.0 * (ll_sat - ll_param))
    df = quad.shape[1]
    p = float(1.0 - chi2.cdf(stat, df))

    return FusionMismatchResult(
        statistic=float(stat),
        p_value=p,
        degrees_of_freedom=df,
        reject=bool(p < alpha),
    )
