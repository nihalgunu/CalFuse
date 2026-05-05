"""Reranker-as-fusion baseline.

Treats the cross-encoder logit as *the* fused score and ignores all other
signals — the standard production setup in systems like ColBERTv2 +
MonoT5 (Khattab & Zaharia, 2020; Nogueira et al., 2020). We include it
as a baseline because it represents the "just use a big reranker"
argument against multi-signal fusion. A post-hoc Platt calibrator is
fitted on the reranker output so that the ECE comparison is
well-defined.

Takes the index of the cross-encoder column at construction. If the
column is absent, falls back to a simple mean of sigmoid-transformed
scores — this corner case is only exercised in synthetic smoke tests.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import BaseFusion


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class RerankerFusion(BaseFusion):
    name = "reranker_fusion"

    def __init__(self, reranker_col: int = -1) -> None:
        self.reranker_col = int(reranker_col)
        self._post_calibrator = None

    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "RerankerFusion":
        X = np.asarray(scores, dtype=np.float64)
        if labels is None or self.reranker_col >= X.shape[1]:
            return self
        from ..calibrators.platt import PlattCalibrator

        raw = X[:, self.reranker_col]
        self._post_calibrator = PlattCalibrator().fit(raw, np.asarray(labels))
        return self

    def fuse(self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None) -> np.ndarray:
        X = np.asarray(scores, dtype=np.float64)
        if self.reranker_col >= X.shape[1]:
            # Fallback path for synthetic tests with no CE column.
            return _sigmoid(X.mean(axis=1))
        raw = X[:, self.reranker_col]
        if self._post_calibrator is not None:
            return self._post_calibrator.transform(raw)
        return _sigmoid(raw)
