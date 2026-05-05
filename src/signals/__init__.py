"""Retrieval signal modules.

Every signal exposes a ``score_pairs(pairs) -> np.ndarray`` method that returns a
real-valued score per ``(query, passage)`` pair. Signals are *not* calibrated —
raw scores may live on arbitrary scales. Calibration is the responsibility of
:mod:`src.calibrators`.
"""

from .base import BaseSignal

__all__ = ["BaseSignal"]
