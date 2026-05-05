"""Distribution-free conformal calibration for retrieval fusion.

This package is the Phase-2+ risk contribution of CalFuse: we build
*distribution-free finite-sample* calibration guarantees on fused
retrieval scores, and we extend the guarantee to an *anytime-valid*
setting where signals are computed sequentially and the stopping time
is data-adaptive. The latter turns calibration into a
computation-budget problem and is --- to our knowledge --- the first
application of e-process / sequential-testing machinery to retrieval
fusion.

The three modules:

* :mod:`venn_abers` --- inductive Venn-Abers predictors (Vovk & Petej,
  2014) on a base fused score. Output: an envelope
  ``[p_lo(x), p_hi(x)]`` such that marginally over the calibration
  split one of ``{p_lo, p_hi}`` is perfectly calibrated in the
  distribution-free sense.
* :mod:`mondrian` --- signal-stratified Mondrian-Venn-Abers: per-
  subgroup envelopes that are conditionally valid within each
  stratum, satisfying a strictly stronger coverage guarantee than
  marginal Venn-Abers.
* :mod:`sequential` --- anytime-valid e-processes constructed from
  calibrated signal likelihood ratios. Given a cost-ordered signal
  sequence (BM25 cheap, cross-encoder expensive), the e-process
  gives a provably-valid stopping rule: compute signals until the
  product e-value crosses ``1/alpha`` (decide relevant) or drops
  below ``alpha`` (decide irrelevant), and the Type-I error is
  controlled uniformly across stopping times by Ville's inequality.

Supporting theory: Theorems 5 and 6 in ``theory/proofs.tex``.
"""

from .mondrian import MondrianVennAbers
from .sequential import EProcess, SequentialFusionDecision, sequential_fusion
from .venn_abers import VennAbersPredictor

__all__ = [
    "EProcess",
    "MondrianVennAbers",
    "SequentialFusionDecision",
    "VennAbersPredictor",
    "sequential_fusion",
]
