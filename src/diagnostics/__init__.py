"""Diagnostic tests for CalFuse failure modes.

The failure-mode taxonomy in ``theory/proofs.tex`` identifies three
regimes in which calibration-aware fusion degrades:

* **Signal dependence** — two signals are not conditionally independent
  given the latent relevance label. Implemented in
  :mod:`signal_dependence`.
* **Calibration drift** — the calibration-fit split is drawn from a
  different distribution than the test split. Implemented in
  :mod:`calibration_drift`.
* **Fusion-rule mismatch** — the true data-generating process does not
  match the parametric form assumed by the fusion rule. Implemented in
  :mod:`fusion_mismatch`.

Each diagnostic returns both a scalar test statistic and a binary flag
at a benchmark-standard significance level (``alpha = 0.05``). The
flag is used by CalFuse to (i) switch between parametric and learned
variants (signal dependence) and (ii) emit warnings downstream
(calibration drift, fusion mismatch).
"""

from .calibration_drift import calibration_drift_test
from .fusion_mismatch import fusion_mismatch_test
from .signal_dependence import signal_dependence_test

__all__ = [
    "calibration_drift_test",
    "fusion_mismatch_test",
    "signal_dependence_test",
]
