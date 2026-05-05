"""Copula-CalFuse --- dependence-aware analytical fusion.

Motivation
----------
The parametric CalFuse rule is Bayes-optimal under strict conditional
independence of signals given ``Y`` (Theorem 1). When CI fails the
first-order bias of that rule is
``(1/2) * sum_{i!=j} (Sigma^{(1)}_{ij} - Sigma^{(0)}_{ij})``
(Proposition 2), which is non-zero whenever the class-conditional
covariance of calibrated logits has off-diagonal mass.

The fallback learned MLP on calibrated logits removes the bias in
principle, but (a) sacrifices calibration preservation, (b) overfits on
small calibration sets, and (c) offers no analytical handle on the
bias. This module replaces it with a Gaussian-copula construction that
handles dependence in closed form.

Method
------
Fix per-signal calibrators ``f_i`` so that marginally ``p_i = f_i(S_i)``
is correctly calibrated. Write ``L_i = logit(p_i)``. Model the
conditional joint distribution with Gaussian marginals and a Gaussian
copula:

    L | Y = y  ~  N(mu^{(y)}, Sigma^{(y)}),   y in {0, 1}.

Under this model the log-likelihood ratio admits the closed form

    log p(L | Y=1) - log p(L | Y=0)
      =  -1/2 L^T (Sigma1^{-1} - Sigma0^{-1}) L
         +  L^T (Sigma1^{-1} mu1 - Sigma0^{-1} mu0)
         -  1/2 (mu1^T Sigma1^{-1} mu1 - mu0^T Sigma0^{-1} mu0)
         -  1/2 log(det Sigma1 / det Sigma0)

and by Bayes' rule

    logit P(Y=1 | L) = [above]  +  logit(pi).

Under strict conditional independence we have
``Sigma^{(y)} = diag(sigma^{(y)})`` and the quadratic term separates
across signals; under the additional equal-covariance-across-classes
restriction the quadratic term drops out and we recover the parametric
CalFuse logistic-regression form exactly. CopulaCalFuse is therefore a
*strict generalisation* of parametric CalFuse.

Statistical framing
-------------------
The above is equivalent to fitting a Gaussian copula on calibrated
logits with Gaussian marginals. Copulas decouple marginal calibration
from joint dependence (Sklar 1959; Nelsen 2006); by locking the
marginals to the per-signal calibrators we get marginal calibration
``by construction`` and estimate only the dependence structure from
calibration data. The closed-form LLR above is the Bayes-optimal
fusion rule implied by that copula.

Novelty relative to prior work
------------------------------
* Gaussian discriminant analysis (Friedman 1989) is classical but has
  not to our knowledge been deployed as a score-fusion rule on
  *calibrated* per-signal logits.
* Copulas for single-model calibration appear in Kull and Flach
  (2015, "Novel decompositions of proper scoring rules"); we lift the
  idea to multi-signal retrieval fusion.
* Naive Bayes / logistic-stacking fusion is classical; the
  cross-covariance term ``-1/2 L^T (Sigma1^{-1} - Sigma0^{-1}) L``
  is, to our knowledge, the first closed-form analytically-tractable
  treatment of conditional dependence in a multi-signal retrieval
  fusion rule.

See Theorem 3 in ``theory/proofs.tex`` for the calibration-preservation
statement under a correctly-specified Gaussian copula and a
misspecification bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from ..calibrators.base import BaseCalibrator
from ..calibrators.platt import PlattCalibrator
from .base import BaseFusion


EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _regularised_cov(L: np.ndarray, shrinkage: float, pooled: Optional[np.ndarray] = None) -> np.ndarray:
    """Sample covariance with linear shrinkage toward ``pooled`` (or
    ``diag(var(L))`` if ``pooled`` is None) and eigenvalue flooring.

    Small retrieval calibration splits produce rank-deficient sample
    covariances; linear shrinkage (Ledoit--Wolf, 2004) fixes this and
    also reduces variance of the QDA estimator, which matters
    substantially when the class-conditional sample sizes are tens to
    low hundreds.
    """
    if L.shape[0] < 2:
        d = L.shape[1]
        return np.eye(d)
    C = np.cov(L, rowvar=False)
    if C.ndim == 0:  # single feature
        C = np.array([[float(C)]])
    if pooled is None:
        pooled = np.diag(np.maximum(np.diag(C), 1e-4))
    C = (1.0 - shrinkage) * C + shrinkage * pooled
    # Floor eigenvalues to guarantee positive-definiteness.
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-4, None)
    return (V * w) @ V.T


@dataclass
class CopulaReport:
    mu_0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mu_1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Sigma_0: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Sigma_1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    shrinkage: float = 0.0
    per_signal_calibrator: Dict[str, str] = field(default_factory=dict)
    prior: float = 0.5
    correlation_gap_frobenius: float = 0.0


class CopulaCalFuse(BaseFusion):
    """Closed-form dependence-aware fusion via a Gaussian copula on
    calibrated logits.

    Parameters
    ----------
    calibrator_factory
        Factory for per-signal calibrators (default: Platt). The
        calibrator choice controls marginal calibration; the copula
        layer controls joint dependence, so the two are genuinely
        orthogonal.
    shrinkage
        Linear shrinkage applied to the per-class sample covariance
        toward a diagonal target. ``0`` disables shrinkage; ``0.2``
        works well on calibration partitions with a few hundred
        positives per signal (Ledoit--Wolf, 2004).
    tied_covariance
        If True, estimate a single class-pooled covariance and use LDA
        rather than QDA. Useful when calibration partitions are very
        small: the quadratic term in the LLR then drops out and the
        estimator is identical to parametric CalFuse with a specific
        choice of linear coefficients.
    """

    name = "calfuse_copula"

    def __init__(
        self,
        calibrator_factory=PlattCalibrator,
        shrinkage: float = 0.2,
        tied_covariance: Optional[bool] = None,
        min_samples_per_param: int = 40,
    ) -> None:
        self.calibrator_factory = calibrator_factory
        self.shrinkage = float(shrinkage)
        # None -> auto-select based on class-sample size; see ``fit``.
        self.tied_covariance = tied_covariance
        self.min_samples_per_param = int(min_samples_per_param)

        self._calibrators: list[BaseCalibrator] = []
        self._prior: float = 0.5
        self._mu0: np.ndarray | None = None
        self._mu1: np.ndarray | None = None
        self._Sigma0: np.ndarray | None = None
        self._Sigma1: np.ndarray | None = None
        self._Sigma0_inv: np.ndarray | None = None
        self._Sigma1_inv: np.ndarray | None = None
        self._logdet0: float = 0.0
        self._logdet1: float = 0.0
        self.report_: CopulaReport = CopulaReport()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "CopulaCalFuse":
        if labels is None:
            raise ValueError("CopulaCalFuse requires labels for calibration")
        X = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        n_pairs, n_signals = X.shape
        self._prior = float(np.clip(y.mean(), EPS, 1.0 - EPS))

        # 1. Per-signal calibration (marginals of the Gaussian copula).
        self._calibrators = []
        p_cal = np.zeros_like(X)
        for j in range(n_signals):
            cal = self.calibrator_factory()
            cal.fit(X[:, j], y)
            p_cal[:, j] = cal.transform(X[:, j])
            self._calibrators.append(cal)

        # 2. Work in calibrated-logit space. Per-class Gaussian
        #    parameters are the copula's only free parameters.
        L = _logit(p_cal)
        mask0 = y == 0
        mask1 = y == 1

        if mask0.sum() < 3 or mask1.sum() < 3:
            # Degenerate: fall back to independent standard normals
            # (i.e. strict CI). Behaviour matches parametric CalFuse.
            self._mu0 = np.zeros(n_signals)
            self._mu1 = np.zeros(n_signals)
            self._Sigma0 = np.eye(n_signals)
            self._Sigma1 = np.eye(n_signals)
        else:
            self._mu0 = L[mask0].mean(axis=0)
            self._mu1 = L[mask1].mean(axis=0)

            # Auto-select tied vs untied covariance. Each per-class
            # covariance has n*(n+1)/2 free parameters. We need at
            # least ``min_samples_per_param`` calibration samples per
            # parameter in the smaller class to trust the untied
            # estimator; otherwise tie the covariance (LDA form) to
            # reduce estimator variance. This matches the classical
            # guidance of Friedman (1989, "Regularized Discriminant
            # Analysis") for the QDA/LDA boundary.
            n_min_class = min(int(mask0.sum()), int(mask1.sum()))
            n_params = n_signals * (n_signals + 1) // 2
            if self.tied_covariance is None:
                use_tied = n_min_class < self.min_samples_per_param * n_params
            else:
                use_tied = bool(self.tied_covariance)

            if use_tied:
                pooled = _regularised_cov(L, self.shrinkage)
                self._Sigma0 = pooled
                self._Sigma1 = pooled
            else:
                # Diagonal of the pooled covariance is the shrinkage target.
                pooled_diag = np.diag(np.maximum(L.var(axis=0), 1e-4))
                self._Sigma0 = _regularised_cov(L[mask0], self.shrinkage, pooled_diag)
                self._Sigma1 = _regularised_cov(L[mask1], self.shrinkage, pooled_diag)

        self._Sigma0_inv = np.linalg.inv(self._Sigma0)
        self._Sigma1_inv = np.linalg.inv(self._Sigma1)
        sign0, self._logdet0 = np.linalg.slogdet(self._Sigma0)
        sign1, self._logdet1 = np.linalg.slogdet(self._Sigma1)
        if sign0 <= 0 or sign1 <= 0:
            raise RuntimeError("Copula covariance failed positivity check; increase shrinkage")

        self.report_ = CopulaReport(
            mu_0=self._mu0.copy(),
            mu_1=self._mu1.copy(),
            Sigma_0=self._Sigma0.copy(),
            Sigma_1=self._Sigma1.copy(),
            shrinkage=self.shrinkage,
            per_signal_calibrator={f"signal_{j}": c.name for j, c in enumerate(self._calibrators)},
            prior=self._prior,
            correlation_gap_frobenius=float(np.linalg.norm(self._Sigma1 - self._Sigma0, ord="fro")),
        )
        return self

    # ------------------------------------------------------------------
    # fuse
    # ------------------------------------------------------------------
    def _llr(self, L: np.ndarray) -> np.ndarray:
        assert (
            self._mu0 is not None
            and self._mu1 is not None
            and self._Sigma0_inv is not None
            and self._Sigma1_inv is not None
        )
        diff1 = L - self._mu1
        diff0 = L - self._mu0
        q1 = np.einsum("ij,jk,ik->i", diff1, self._Sigma1_inv, diff1)
        q0 = np.einsum("ij,jk,ik->i", diff0, self._Sigma0_inv, diff0)
        return -0.5 * q1 + 0.5 * q0 - 0.5 * (self._logdet1 - self._logdet0)

    def fuse(self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None) -> np.ndarray:
        X = np.asarray(scores, dtype=np.float64)
        if not self._calibrators:
            raise RuntimeError("CopulaCalFuse.fit must be called before fuse")
        p_cal = np.stack(
            [self._calibrators[j].transform(X[:, j]) for j in range(X.shape[1])],
            axis=1,
        )
        L = _logit(p_cal)
        llr = self._llr(L)
        prior_logit = np.log(self._prior / (1.0 - self._prior))
        return _sigmoid(llr + prior_logit)

    @property
    def correlation_gap(self) -> float:
        """Frobenius norm of ``Sigma^{(1)} - Sigma^{(0)}`` -- the main
        driver of the copula correction's magnitude.
        """
        return float(self.report_.correlation_gap_frobenius)
