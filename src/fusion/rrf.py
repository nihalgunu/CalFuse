"""Reciprocal Rank Fusion (Cormack, Clarke, Büttcher, 2009).

RRF is the field-standard fusion baseline. For each query and each
signal, pairs are ranked by signal score and assigned a reciprocal rank
``1 / (k + rank)``. The RRF score is the sum of reciprocal ranks across
signals. RRF is attractive because it requires no training and no score
normalisation.

RRF's *calibration* properties are however poor. The RRF score lives on
a bounded rank-based scale with a heavy-tailed distribution whose
relationship to the true probability of relevance depends arbitrarily on
(i) the candidate pool size per query, (ii) the distribution of signal
score ties, and (iii) the number of fused signals. To evaluate RRF on
ECE we therefore fit a Platt calibrator on the RRF output itself during
``fit`` — this gives RRF the benefit of post-hoc calibration and isolates
the claim that *calibrated fusion* beats *fusion + post-hoc calibration*.

See Section 4.1 of the paper for the argument that even post-hoc
calibration of RRF cannot recover a proper calibrated estimator in
general, because the rank-to-probability map is query-dependent.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import BaseFusion


class RRFFusion(BaseFusion):
    name = "rrf"

    def __init__(self, k: float = 60.0) -> None:
        self.k = float(k)
        self._post_calibrator = None

    def _rrf_raw(self, scores: np.ndarray, query_ids: Sequence[str]) -> np.ndarray:
        """Compute the raw RRF sum across signals, grouping by query."""
        n_pairs, n_signals = scores.shape
        rrf = np.zeros(n_pairs, dtype=np.float64)

        # Group row indices by query.
        q_groups: dict[str, list[int]] = {}
        for i, qid in enumerate(query_ids):
            q_groups.setdefault(qid, []).append(i)

        for idxs in q_groups.values():
            idxs = np.array(idxs, dtype=np.int64)
            for j in range(n_signals):
                sub = scores[idxs, j]
                # ``argsort(-sub)`` gives index order from highest score to
                # lowest; ``ranks[i]`` is the rank of row idxs[i].
                order = np.argsort(-sub, kind="mergesort")
                ranks = np.empty_like(order)
                ranks[order] = np.arange(len(order))
                rrf[idxs] += 1.0 / (self.k + ranks)
        return rrf

    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "RRFFusion":
        if query_ids is None:
            raise ValueError("RRFFusion.fit requires query_ids")
        raw = self._rrf_raw(scores, query_ids)
        if labels is not None:
            # Post-hoc Platt on the RRF score so we can report ECE.
            from ..calibrators.platt import PlattCalibrator

            self._post_calibrator = PlattCalibrator().fit(raw, np.asarray(labels))
        return self

    def fuse(self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None) -> np.ndarray:
        if query_ids is None:
            raise ValueError("RRFFusion.fuse requires query_ids")
        raw = self._rrf_raw(scores, query_ids)
        if self._post_calibrator is not None:
            return self._post_calibrator.transform(raw)
        # Normalise to [0,1] via min-max so the unfitted call still returns
        # something in probability range (useful for smoke tests).
        if raw.size == 0:
            return raw
        lo, hi = float(raw.min()), float(raw.max())
        if hi - lo < 1e-12:
            return np.full_like(raw, 0.5)
        return (raw - lo) / (hi - lo)
