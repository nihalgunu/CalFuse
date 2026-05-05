"""Base interface for calibrators.

Every calibrator must implement ``fit(scores, labels)`` and
``transform(scores) -> probabilities``. Calibrators must be monotone in the
raw score — if not, the benchmark's reliability diagrams become hard to
interpret and the calibration preservation theorem (see ``theory/proofs``)
ceases to hold. Current implementations are all monotone by construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseCalibrator(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "BaseCalibrator":
        raise NotImplementedError

    @abstractmethod
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities in (0, 1)."""
        raise NotImplementedError

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(scores, labels).transform(scores)
