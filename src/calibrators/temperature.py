"""Temperature-scaling calibrator (Guo et al., 2017).

Originally proposed for multi-class softmax outputs, temperature scaling
on binary logits reduces to fitting a single scalar ``T > 0`` such that
the calibrated probability is ``sigmoid(logit / T)``. Because the
intercept is not allowed to move, temperature scaling is strictly *less
expressive* than Platt scaling; we include it as a baseline because it
is the most widely used calibrator for neural-network outputs in the
literature, and the Phase 3 ablations need a comparator that matches
common practice.

We fit ``T`` by one-dimensional minimisation of NLL on the calibration
split via ``scipy.optimize.minimize_scalar`` (Brent's method). The
bracket is set to ``[0.05, 20]`` which covers every case we have seen
on BEIR cross-encoder logits.
"""

from __future__ import annotations

import numpy as np

from .base import BaseCalibrator


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class TemperatureCalibrator(BaseCalibrator):
    name = "temperature"

    def __init__(self, bracket: tuple[float, float] = (0.05, 20.0)) -> None:
        self.bracket = bracket
        self._T: float = 1.0
        self._fitted: bool = False

    def _nll(self, T: float, logits: np.ndarray, labels: np.ndarray) -> float:
        if T <= 0:
            return np.inf
        p = _sigmoid(logits / T)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "TemperatureCalibrator":
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if s.size == 0 or np.all(y == y[0]):
            self._T = 1.0
            self._fitted = True
            return self
        try:
            from scipy.optimize import minimize_scalar  # type: ignore

            res = minimize_scalar(
                lambda T: self._nll(T, s, y),
                bracket=self.bracket,
                method="brent",
            )
            self._T = float(res.x) if res.x > 0 else 1.0
        except Exception:  # pragma: no cover
            # Grid-search fallback.
            grid = np.geomspace(self.bracket[0], self.bracket[1], 64)
            losses = np.array([self._nll(T, s, y) for T in grid])
            self._T = float(grid[int(np.argmin(losses))])
        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TemperatureCalibrator.fit must be called before transform")
        return _sigmoid(np.asarray(scores, dtype=np.float64) / self._T)
