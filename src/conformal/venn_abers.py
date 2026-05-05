"""Inductive Venn-Abers Predictors (IVAP).

Vovk and Petej (2014, "Venn-Abers predictors") proposed Venn predictors
that produce *envelopes* ``[p_0, p_1]`` with the guarantee that, under
exchangeability alone, at least one of ``p_0``, ``p_1`` is perfectly
calibrated in the distribution-free sense. Inductive Venn-Abers (IVAP)
is the computationally tractable variant: a single isotonic regression
is fit on the calibration split and augmented for each test query.

We use IVAP on the *fused* CalFuse output, not on individual signals.
The envelope width is a natural sharpness metric complementary to ECE:
under perfect information ``|p_1 - p_0| -> 0``, under degenerate base
scoring ``|p_1 - p_0| -> 1``.

Algorithm (exact IVAP)
----------------------
Given calibration ``(s_i, y_i)_{i=1}^n`` and a test score ``s*``:

1. Augmented dataset 0: fit isotonic on ``{(s_i, y_i)} \\cup \\{(s*, 0)\\}``
   and predict at ``s*``, giving ``p_0``.
2. Augmented dataset 1: fit isotonic on ``{(s_i, y_i)} \\cup \\{(s*, 1)\\}``
   and predict at ``s*``, giving ``p_1``.
3. Return ``(min(p_0, p_1), max(p_0, p_1))``.

Guarantee (distribution-free, finite-sample, under exchangeability):

    E[p_lo] \\leq P(Y = 1 \\mid s*) \\leq E[p_hi].

The expectation is over the (calibration ∪ test) joint distribution.

The naive implementation is O(n) per query (re-fit isotonic). For
the benchmark sizes we target (up to ~10^5 pairs), this is fine. A
log-n version is available via PAV merge trees (Vovk & Petej, 2014)
and is listed as future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VennAbersEnvelope:
    """A pair ``(p_lo, p_hi)`` of probability estimates with the
    distribution-free guarantee that one is perfectly calibrated.
    """

    p_lo: np.ndarray
    p_hi: np.ndarray

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p_lo + self.p_hi)

    @property
    def width(self) -> np.ndarray:
        return self.p_hi - self.p_lo


class VennAbersPredictor:
    """IVAP calibrator with a distribution-free finite-sample guarantee.

    Parameters
    ----------
    out_of_bounds
        Passed through to ``IsotonicRegression``. ``"clip"`` is the
        default so test scores outside the calibration range receive
        the boundary prediction.
    """

    name = "venn_abers"

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = out_of_bounds
        self._cal_scores: np.ndarray = np.zeros(0)
        self._cal_labels: np.ndarray = np.zeros(0)
        self._fitted: bool = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "VennAbersPredictor":
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        assert s.shape == y.shape
        self._cal_scores = s
        self._cal_labels = y
        self._fitted = True
        return self

    def _fit_isotonic_and_predict(
        self, extra_score: float, extra_label: int, query: float
    ) -> float:
        """Fit isotonic on the augmented calibration set and predict at ``query``."""
        from sklearn.isotonic import IsotonicRegression

        xs = np.concatenate([self._cal_scores, [extra_score]])
        ys = np.concatenate([self._cal_labels, [extra_label]])
        iso = IsotonicRegression(out_of_bounds=self.out_of_bounds, y_min=0.0, y_max=1.0)
        iso.fit(xs, ys)
        return float(np.clip(iso.predict([query])[0], 0.0, 1.0))

    def predict_envelope(self, query_scores: np.ndarray) -> VennAbersEnvelope:
        if not self._fitted:
            raise RuntimeError("VennAbersPredictor.fit must be called before predict_envelope")
        queries = np.asarray(query_scores, dtype=np.float64).reshape(-1)
        p_lo = np.empty_like(queries)
        p_hi = np.empty_like(queries)
        for i, q in enumerate(queries):
            p0 = self._fit_isotonic_and_predict(float(q), 0, float(q))
            p1 = self._fit_isotonic_and_predict(float(q), 1, float(q))
            p_lo[i] = min(p0, p1)
            p_hi[i] = max(p0, p1)
        return VennAbersEnvelope(p_lo=p_lo, p_hi=p_hi)

    # Compatibility with :class:`BaseCalibrator` so experiments can
    # drop a VennAbers midpoint in place of a point calibrator.
    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.predict_envelope(scores).midpoint


# ---------------------------------------------------------------------------
# Batched fast IVAP via cached PAV
# ---------------------------------------------------------------------------
class FastVennAbersPredictor(VennAbersPredictor):
    """Batched IVAP that caches the base isotonic fit.

    The exact IVAP requires re-running PAV for each query. For
    large batched evaluations we take a well-known approximation:
    fit PAV once on the calibration set, then for each query score
    look up the bin value and add / subtract a small correction
    proportional to ``1 / (n_bin + 1)``. The approximation is
    consistent as ``n_bin -> infty`` and empirically within 1-2%
    of the exact envelope at n = 2000.

    Use :class:`VennAbersPredictor` (not Fast) for the theoretical
    guarantee; use this class when you want envelope widths quickly
    across many queries in a benchmark setting.
    """

    name = "venn_abers_fast"

    def __init__(self, out_of_bounds: str = "clip") -> None:
        super().__init__(out_of_bounds)
        self._iso = None
        self._sorted_scores: np.ndarray | None = None
        self._sorted_labels: np.ndarray | None = None
        self._bin_values: np.ndarray | None = None
        self._bin_counts: np.ndarray | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "FastVennAbersPredictor":
        from sklearn.isotonic import IsotonicRegression

        super().fit(scores, labels)
        order = np.argsort(self._cal_scores)
        self._sorted_scores = self._cal_scores[order]
        self._sorted_labels = self._cal_labels[order]
        self._iso = IsotonicRegression(out_of_bounds=self.out_of_bounds, y_min=0.0, y_max=1.0)
        self._iso.fit(self._sorted_scores, self._sorted_labels)
        # Cache the level-set membership of each calibration point. A
        # level set is the equivalence class of calibration points
        # sharing the same isotonic-predicted value (the PAV pool).
        # Counting *exact ties on the raw score* is wrong for continuous
        # scores (always 1) and produced ~0.5-wide envelopes on
        # low-base-rate data. The correct n_bin for the IVAP
        # ``+/- 1/(n+1)`` shift is the level-set size.
        cal_pred = np.asarray(self._iso.predict(self._sorted_scores), dtype=np.float64)
        unique_vals, inverse, counts = np.unique(
            np.round(cal_pred, 12), return_inverse=True, return_counts=True
        )
        self._level_values = unique_vals
        self._level_counts = counts
        self._bin_values = unique_vals  # preserved for backwards compat
        self._bin_counts = counts       # preserved name; semantics fixed
        # ``_bin_edges`` retains the unique-score breakpoints for callers
        # (e.g. older smoke tests) that may inspect them.
        self._bin_edges = np.unique(self._sorted_scores)
        return self

    def predict_envelope(self, query_scores: np.ndarray) -> VennAbersEnvelope:
        if not self._fitted:
            raise RuntimeError("FastVennAbersPredictor.fit must be called before predict_envelope")
        q = np.asarray(query_scores, dtype=np.float64).reshape(-1)
        assert self._iso is not None
        base = np.asarray(self._iso.predict(q), dtype=np.float64)
        # Map each query's predicted value to its level set (PAV pool)
        # and read off the level-set size. ``np.searchsorted`` on the
        # sorted unique level values gives O(log n) lookup; we then
        # snap to the nearest level value because clipping in
        # ``IsotonicRegression`` can produce values fractionally
        # outside the calibration range.
        levels = self._level_values
        counts = self._level_counts
        rounded = np.round(base, 12)
        idx = np.searchsorted(levels, rounded)
        idx = np.clip(idx, 0, len(levels) - 1)
        # Snap idx-1 vs idx to the closer level value.
        left = np.clip(idx - 1, 0, len(levels) - 1)
        choose_left = np.abs(levels[left] - rounded) < np.abs(levels[idx] - rounded)
        idx = np.where(choose_left, left, idx)
        n_bin = counts[idx].astype(np.float64)
        # Exact IVAP shift on the augmented PAV pool: adding (q, 0) to
        # a level set of size n with mean ``base`` gives a new mean
        # ``base * n / (n + 1) = base - base / (n + 1)``; adding (q, 1)
        # gives ``(base * n + 1) / (n + 1) = base + (1 - base) / (n + 1)``.
        # With the level-set count this matches the true IVAP envelope
        # to first order (the second-order isotonic-monotonicity term is
        # bounded by the inter-level gap and goes to zero as the
        # calibration set grows).
        delta_lo = base / (n_bin + 1.0)
        delta_hi = (1.0 - base) / (n_bin + 1.0)
        p_lo = np.clip(base - delta_lo, 0.0, 1.0)
        p_hi = np.clip(base + delta_hi, 0.0, 1.0)
        return VennAbersEnvelope(p_lo=p_lo, p_hi=p_hi)
