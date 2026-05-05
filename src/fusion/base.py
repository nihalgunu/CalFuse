"""Base interface for fusion rules.

Every fusion rule implements:

* ``fit(scores, labels, query_ids) -> self``: calibration-phase fit. Some
  rules (RRF, uncalibrated reranker) have no trainable parameters and the
  implementation is a no-op; we keep the hook so that the evaluate
  pipeline does not need to special-case them.
* ``fuse(scores, query_ids) -> probabilities``: return a composite
  probability per row, same length as ``scores.shape[0]``.

We expose ``query_ids`` because rank-based rules (RRF) must group pairs
by query before converting scores to ranks. Pair-level rules ignore the
argument.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class BaseFusion(ABC):
    name: str = "base"

    @abstractmethod
    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "BaseFusion":
        raise NotImplementedError

    @abstractmethod
    def fuse(
        self,
        scores: np.ndarray,
        query_ids: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        raise NotImplementedError
