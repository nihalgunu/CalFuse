"""CalFuse — calibration-aware fusion of heterogeneous retrieval signals.

Core idea (see ``theory/proofs.tex`` for the formal development):

1. Calibrate each signal ``S_i`` independently on the calibration split,
   producing ``p_i(x) = P(Y=1 | S_i = s_i)`` where ``Y`` is the latent
   relevance variable.
2. Under conditional independence ``S_i ⟂ S_j | Y`` the Bayes-optimal
   composite log-odds is the weighted sum of per-signal log-odds minus a
   prior-correction term (Theorem 1):

       logit P(Y=1 | S_1, ..., S_n) =
           sum_i w_i · logit(p_i)  -  c · logit(π)

   with ``w_i = 1`` and ``c = n - 1`` under strict independence. We let
   ``w_i`` and ``c`` be *learned* scalars so the method gracefully
   absorbs mild miscalibration and mild dependence. Because the
   transform is affine in the per-signal logits, it is itself a
   calibrated estimator when the per-signal calibrators are well
   specified and independence holds.
3. When a dependence diagnostic (see
   :mod:`src.diagnostics.signal_dependence`) indicates that conditional
   independence is strongly violated, CalFuse switches to a *learned*
   variant: a two-layer MLP over the per-signal logits with L2
   regularisation that shrinks solutions toward the parametric form.
   This gives up strict calibration preservation but — per Theorem 2
   and the accompanying asymptotic bias analysis — empirically recovers
   calibration on realistic dependent-signal tiers.

Method selection is done automatically at fit time. The choice, the
diagnostic statistics, and the per-signal calibrator identities are all
exposed via :attr:`CalFuseFusion.report_` so that experimental tables
can be built without re-running the fit.

Citations in the header: Platt (1999), Zadrozny & Elkan (2002),
Niculescu-Mizil & Caruana (2005), Guo et al. (2017), Gneiting & Raftery
(2007), Dawid (1982), and for retrieval-side priors Cormack et al.
(2009) and Nogueira & Cho (2019). Full bibliography in ``paper/main.tex``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from ..calibrators.base import BaseCalibrator
from ..calibrators.isotonic import IsotonicCalibrator
from ..calibrators.platt import PlattCalibrator
from .base import BaseFusion


EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class CalFuseReport:
    """Post-fit diagnostics exposed for experimental tables."""

    mode: str = "parametric"  # or "learned"
    dependence_score: float = 0.0
    dependence_threshold: float = 0.3
    prior: float = 0.5
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    prior_correction: float = 0.0
    per_signal_calibrator: Dict[str, str] = field(default_factory=dict)


class CalFuseFusion(BaseFusion):
    """Calibration-aware fusion.

    Parameters
    ----------
    calibrator_factory
        Callable returning a fresh :class:`BaseCalibrator` per signal.
        Defaults to Platt scaling (Platt, 1999), the simplest calibrator
        with provable consistency under the logistic-link assumption.
    dependence_threshold
        Maximum permissible absolute partial correlation between any two
        calibrated per-signal logits, conditional on the label, before
        the learned variant is preferred over the parametric one. The
        default ``0.3`` is informed by the sensitivity study in
        Phase 3 ablations (larger thresholds keep the parametric form
        too long; smaller ones trigger the learned variant unnecessarily).
    """

    name = "calfuse"

    def __init__(
        self,
        calibrator_factory=PlattCalibrator,
        dependence_threshold: float = 0.3,
        mlp_hidden: int = 16,
        mlp_alpha: float = 1e-3,
        mlp_max_iter: int = 2000,
        force_mode: Optional[str] = None,
    ) -> None:
        self.calibrator_factory = calibrator_factory
        self.dependence_threshold = float(dependence_threshold)
        self.mlp_hidden = int(mlp_hidden)
        self.mlp_alpha = float(mlp_alpha)
        self.mlp_max_iter = int(mlp_max_iter)
        self.force_mode = force_mode  # {"parametric", "learned", None}

        self._calibrators: list[BaseCalibrator] = []
        self._weights: np.ndarray | None = None
        self._prior_correction: float = 0.0
        self._prior: float = 0.5
        self._mode: str = "parametric"
        self._mlp = None
        self._mlp_mean: np.ndarray | None = None
        self._mlp_std: np.ndarray | None = None
        self.report_: CalFuseReport = CalFuseReport()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _conditional_partial_correlation(logits: np.ndarray, y: np.ndarray) -> float:
        """Max ``|corr(logit_i, logit_j | Y=k)|`` over i<j and k in {0,1}.

        Returns 0 if either class has fewer than two samples.
        """
        n, d = logits.shape
        max_abs = 0.0
        for k in (0, 1):
            mask = y == k
            if mask.sum() < 2:
                continue
            sub = logits[mask]
            if sub.shape[0] < 2:
                continue
            stds = sub.std(axis=0)
            if np.any(stds < 1e-9):
                continue
            C = np.corrcoef(sub, rowvar=False)
            if np.isnan(C).any():
                continue
            for i in range(d):
                for j in range(i + 1, d):
                    max_abs = max(max_abs, float(abs(C[i, j])))
        return max_abs

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "CalFuseFusion":
        if labels is None:
            raise ValueError("CalFuseFusion requires labels for calibration")
        X = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        n_pairs, n_signals = X.shape

        # Empirical relevance prior from the calibration split.
        self._prior = float(np.clip(y.mean(), EPS, 1.0 - EPS))

        # 1. Per-signal calibration.
        self._calibrators = []
        p_cal = np.zeros_like(X)
        for j in range(n_signals):
            cal = self.calibrator_factory()
            cal.fit(X[:, j], y)
            p_cal[:, j] = cal.transform(X[:, j])
            self._calibrators.append(cal)

        # 2. Diagnose dependence in the calibrated logit space.
        logit_cal = _logit(p_cal)
        dep = self._conditional_partial_correlation(logit_cal, y)

        use_learned = (
            (self.force_mode == "learned")
            or (self.force_mode != "parametric" and dep > self.dependence_threshold)
        )

        if not use_learned:
            # 3a. Parametric fusion: logistic regression on calibrated
            #     logits with an intercept that absorbs the prior
            #     correction. Fitting the weights (rather than fixing
            #     them at 1) lets CalFuse absorb residual per-signal
            #     miscalibration that the independent calibrator missed.
            self._fit_parametric(logit_cal, y)
            self._mode = "parametric"
        else:
            self._fit_learned(logit_cal, y)
            self._mode = "learned"

        # 4. Pack diagnostic report.
        self.report_ = CalFuseReport(
            mode=self._mode,
            dependence_score=dep,
            dependence_threshold=self.dependence_threshold,
            prior=self._prior,
            weights=np.array(self._weights) if self._weights is not None else np.zeros(0),
            prior_correction=float(self._prior_correction),
            per_signal_calibrator={
                f"signal_{j}": c.name for j, c in enumerate(self._calibrators)
            },
        )
        return self

    def _fit_parametric(self, logit_cal: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000)
        # Intercept is free. The theoretical form is
        #   logit(P) = sum_i w_i * logit(p_i) + intercept
        # where intercept plays the role of ``- c * logit(prior)`` under
        # strict independence. We recover c from the fitted intercept.
        if np.unique(y).size < 2:
            # Degenerate: default to uniform weights and prior intercept.
            self._weights = np.ones(logit_cal.shape[1], dtype=np.float64)
            self._prior_correction = (logit_cal.shape[1] - 1.0) * np.log(self._prior / (1.0 - self._prior))
            return
        lr.fit(logit_cal, y)
        self._weights = lr.coef_[0].astype(np.float64)
        self._prior_correction = -float(lr.intercept_[0])

    def _fit_learned(self, logit_cal: np.ndarray, y: np.ndarray) -> None:
        try:
            from sklearn.neural_network import MLPClassifier

            # Standardise so L-BFGS converges in a reasonable number of
            # iterations regardless of the per-signal logit scale.
            self._mlp_mean = logit_cal.mean(axis=0)
            self._mlp_std = logit_cal.std(axis=0)
            self._mlp_std[self._mlp_std < 1e-9] = 1.0
            Xs = (logit_cal - self._mlp_mean) / self._mlp_std

            mlp = MLPClassifier(
                hidden_layer_sizes=(self.mlp_hidden,),
                activation="tanh",
                solver="lbfgs",
                alpha=self.mlp_alpha,
                max_iter=self.mlp_max_iter,
                random_state=0,
            )
            mlp.fit(Xs, y)
            self._mlp = mlp
        except Exception:  # pragma: no cover
            # Fallback to parametric if sklearn's MLP is unavailable.
            self._fit_parametric(logit_cal, y)
            self._mode = "parametric"

    # ------------------------------------------------------------------
    # fuse
    # ------------------------------------------------------------------
    def fuse(self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None) -> np.ndarray:
        X = np.asarray(scores, dtype=np.float64)
        if not self._calibrators:
            raise RuntimeError("CalFuseFusion.fit must be called before fuse")
        # 1. Apply per-signal calibrators.
        p_cal = np.stack(
            [self._calibrators[j].transform(X[:, j]) for j in range(X.shape[1])],
            axis=1,
        )
        logit_cal = _logit(p_cal)
        if self._mode == "learned" and self._mlp is not None:
            assert self._mlp_mean is not None and self._mlp_std is not None
            Xs = (logit_cal - self._mlp_mean) / self._mlp_std
            return np.clip(self._mlp.predict_proba(Xs)[:, 1], EPS, 1.0 - EPS)
        assert self._weights is not None
        z = logit_cal @ self._weights - self._prior_correction
        return _sigmoid(z)

    # ------------------------------------------------------------------
    # inspection helpers
    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    def calibrated_logits(self, scores: np.ndarray) -> np.ndarray:
        X = np.asarray(scores, dtype=np.float64)
        p_cal = np.stack(
            [self._calibrators[j].transform(X[:, j]) for j in range(X.shape[1])],
            axis=1,
        )
        return _logit(p_cal)
