"""Single-signal + Platt baselines for SOTA reference points on BEIR.

Each canonical BEIR paper reports numbers for the individual retrievers
alone: BM25 (Robertson and Zaragoza, 2009), BGE (Xiao et al., 2023), E5
(Wang et al., 2022), and a cross-encoder reranker such as
``cross-encoder/ms-marco-MiniLM-L-12-v2`` (Reimers and Gurevych, 2019).
These are the "best single method" SOTA reference points the retrieval
community compares against; reporting them alongside our fusion variants
makes it clear that the headline numbers are not an artefact of picking
a weak baseline.

The fusion rule is trivial: take one column of the signal matrix and run
Platt calibration on it. Monotone in the raw score, so NDCG equals what
the underlying retriever would score on its own; ECE reflects the
strength of single-signal calibration relative to fusion.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..calibrators.platt import PlattCalibrator
from .base import BaseFusion


class SingleSignalFusion(BaseFusion):
    """Column-selector + Platt calibration — the SOTA single-retriever baseline."""

    def __init__(self, col_idx: int, name: str = "single_signal") -> None:
        self.col_idx = int(col_idx)
        self.name = name
        self._calibrator: PlattCalibrator | None = None

    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "SingleSignalFusion":
        if labels is None:
            raise ValueError("SingleSignalFusion requires labels for Platt calibration")
        X = np.asarray(scores, dtype=np.float64)
        s = X[:, self.col_idx]
        self._calibrator = PlattCalibrator().fit(s, np.asarray(labels))
        return self

    def fuse(
        self,
        scores: np.ndarray,
        query_ids: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if self._calibrator is None:
            raise RuntimeError("SingleSignalFusion.fit must be called before fuse")
        X = np.asarray(scores, dtype=np.float64)
        return self._calibrator.transform(X[:, self.col_idx])
