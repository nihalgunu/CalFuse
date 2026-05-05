"""Isotonic-regression calibrator (Zadrozny & Elkan, 2002).

Isotonic regression relaxes the parametric logistic-shape assumption of
Platt scaling at the cost of a higher effective parameter count (roughly
O(n) breakpoints). On benchmarks with enough calibration data isotonic
typically wins the ECE race (Niculescu-Mizil & Caruana, 2005); on small
calibration partitions it overfits, which is why we offer both as
alternatives and study the trade-off in the Phase 3 ablations.

We use scikit-learn's ``IsotonicRegression`` with ``out_of_bounds="clip"``
so that calibrated probabilities remain well-defined when test scores
fall outside the fit range. When sklearn is unavailable we fall back to
a small Pool-Adjacent-Violators implementation — slower but dependency-free.
"""

from __future__ import annotations

import numpy as np

from .base import BaseCalibrator


def _pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-Adjacent-Violators algorithm (Ayer et al., 1955).

    Returns the sorted unique x values and the monotone-fit y values at
    those points. O(n) after the sort.
    """
    order = np.argsort(x, kind="mergesort")
    xs = x[order].astype(np.float64)
    ys = y[order].astype(np.float64)
    w = np.ones_like(ys)
    # Pool until monotone non-decreasing.
    i = 0
    while i < len(ys) - 1:
        if ys[i] > ys[i + 1]:
            total_w = w[i] + w[i + 1]
            pooled = (w[i] * ys[i] + w[i + 1] * ys[i + 1]) / total_w
            ys[i] = pooled
            w[i] = total_w
            ys = np.delete(ys, i + 1)
            xs_new_center = xs[i]  # keep leftmost break point
            xs = np.delete(xs, i + 1)
            w = np.delete(w, i + 1)
            if i > 0:
                i -= 1
            _ = xs_new_center
        else:
            i += 1
    return xs, ys


class IsotonicCalibrator(BaseCalibrator):
    name = "isotonic"

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = out_of_bounds
        self._xs: np.ndarray | None = None
        self._ys: np.ndarray | None = None
        self._model = None
        self._fitted: bool = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if s.size == 0:
            self._xs, self._ys = np.array([0.0]), np.array([0.5])
            self._fitted = True
            return self
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore

            iso = IsotonicRegression(out_of_bounds=self.out_of_bounds, y_min=1e-6, y_max=1 - 1e-6)
            iso.fit(s, y)
            self._model = iso
        except Exception:  # pragma: no cover
            xs, ys = _pav(s, y)
            self._xs = xs
            self._ys = np.clip(ys, 1e-6, 1 - 1e-6)
        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator.fit must be called before transform")
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        if self._model is not None:
            p = self._model.predict(s)
            return np.clip(p, 1e-6, 1 - 1e-6)
        assert self._xs is not None and self._ys is not None
        # Piecewise-constant interpolation with clipping at the edges.
        idx = np.searchsorted(self._xs, s, side="right") - 1
        idx = np.clip(idx, 0, len(self._ys) - 1)
        return self._ys[idx]
