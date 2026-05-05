"""Platt scaling calibrator (Platt, 1999).

Fits a one-dimensional logistic regression ``sigmoid(a * s + b)`` on raw
scores ``s`` against binary relevance labels. We use scikit-learn's
``LogisticRegression`` with a weak L2 regulariser (``C=1e6``) — the
regulariser is present only to guarantee numerical convergence when
scores are perfectly separable (which occasionally happens on small
calibration partitions). Platt scaling is our default per-signal
calibrator because:

* It is the simplest calibrator that consistently outperforms the
  uncalibrated sigmoid on held-out data (Niculescu-Mizil & Caruana,
  2005).
* It is monotone in the raw score, preserving the ranking and therefore
  NDCG — an important invariance under the benchmark's secondary metric.
* Its two-parameter form is cheap to refit in cross-validation loops,
  which matters for the calibration-drift diagnostic in
  :mod:`src.diagnostics.calibration_drift`.
"""

from __future__ import annotations

import numpy as np

from .base import BaseCalibrator


class PlattCalibrator(BaseCalibrator):
    name = "platt"

    def __init__(self, C: float = 1e6) -> None:
        self.C = float(C)
        self._a: float = 1.0
        self._b: float = 0.0
        self._fitted: bool = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        # Degenerate-label guards: the benchmark builder should prevent
        # single-class splits, but we defend in depth for CI runs that
        # exercise tiny synthetic splits.
        if labels.size == 0:
            self._a, self._b = 1.0, 0.0
            self._fitted = True
            return self
        if np.all(labels == labels[0]):
            # Single-class calibration set. Fall back to a flat predictor
            # centred on the empirical rate (which is 0 or 1, offset by a
            # small epsilon so sigmoid stays inside (0, 1)).
            eps = 1e-3
            p = 1.0 - eps if labels[0] == 1 else eps
            self._a = 0.0
            self._b = float(np.log(p / (1.0 - p)))
            self._fitted = True
            return self

        try:  # Prefer scikit-learn when available.
            from sklearn.linear_model import LogisticRegression  # type: ignore

            lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000)
            lr.fit(scores, labels)
            self._a = float(lr.coef_[0, 0])
            self._b = float(lr.intercept_[0])
        except Exception:  # pragma: no cover - numpy fallback
            self._a, self._b = self._fit_numpy(scores.reshape(-1), labels)

        self._fitted = True
        return self

    @staticmethod
    def _fit_numpy(s: np.ndarray, y: np.ndarray, n_iter: int = 200, lr: float = 0.05) -> tuple[float, float]:
        """Gradient-descent fallback logistic fit (used when sklearn absent)."""
        a, b = 0.0, 0.0
        for _ in range(n_iter):
            z = a * s + b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_a = np.mean((p - y) * s)
            grad_b = np.mean(p - y)
            a -= lr * grad_a
            b -= lr * grad_b
        return float(a), float(b)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PlattCalibrator.fit must be called before transform")
        s = np.asarray(scores, dtype=np.float64)
        z = self._a * s + self._b
        # Clip logits for numerical stability.
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))
