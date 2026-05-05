"""Learned-linear fusion baseline.

Fits a logistic regression over the raw signal scores against calibration
labels. The resulting sigmoid *is* a probability, so the composite score
is already on the probability scale. This is the strongest classical
baseline that does not pre-calibrate individual signals: it pools all
score information in a single discriminative model.

Why include it when it is a special case of CalFuse with no per-signal
calibrator? Because "a logistic regression on raw scores" is what most
retrieval practitioners actually do when they want a fused probability.
It makes the comparison concrete and the marginal-improvement numbers
directly interpretable.

Implementation note: we standardise raw scores column-wise (using
calibration-set mean and standard deviation) before fitting. Without
standardisation the logistic's intercept absorbs scale differences
between BM25 (unbounded) and cosine similarity (in ``[-1, 1]``), which
hurts optimisation. Standardisation parameters are frozen at fit time.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import BaseFusion


class LinearLearnedFusion(BaseFusion):
    name = "linear_learned"

    def __init__(self, C: float = 1.0) -> None:
        self.C = float(C)
        self._model = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        return (X - self._mean) / self._std

    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "LinearLearnedFusion":
        if labels is None:
            raise ValueError("LinearLearnedFusion requires labels")
        X = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)

        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-9] = 1.0
        Xs = self._standardise(X)

        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000)
        self._model.fit(Xs, y)
        return self

    def fuse(self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LinearLearnedFusion.fit must be called before fuse")
        X = np.asarray(scores, dtype=np.float64)
        Xs = self._standardise(X)
        return self._model.predict_proba(Xs)[:, 1]
