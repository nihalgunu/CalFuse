"""Per-signal calibrators.

A calibrator is any estimator that takes raw scores and calibration-fit
labels and produces a function ``score -> probability`` such that, on data
drawn from the calibration distribution, the output matches the true
conditional probability of relevance (Dawid, 1982; Gneiting & Raftery, 2007).

CalFuse always calibrates each signal *independently* before fusion; the
fusion rule operates in probability space. See the companion proof in
``theory/proofs.tex`` for why independent calibration is the right layering.
"""

from .base import BaseCalibrator
from .isotonic import IsotonicCalibrator
from .learned_calibrator import LearnedCalibrator
from .platt import PlattCalibrator
from .temperature import TemperatureCalibrator

__all__ = [
    "BaseCalibrator",
    "IsotonicCalibrator",
    "LearnedCalibrator",
    "PlattCalibrator",
    "TemperatureCalibrator",
]
