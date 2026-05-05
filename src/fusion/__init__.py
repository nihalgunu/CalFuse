"""Fusion rules.

A fusion rule takes a per-signal matrix of scores (shape
``(n_pairs, n_signals)``) and returns a composite score per pair. Some
fusion rules only return ranks (RRF); others return arbitrary real scores
(learned linear); CalFuse returns *probabilities*.

The benchmark protocol compares fusion rules on ECE of the composite
score when interpreted as a probability of relevance. Rules that do not
natively output probabilities are converted to probabilities by fitting a
post-hoc calibrator on the calibration split — this keeps the comparison
head-to-head on the calibration axis.
"""

from .base import BaseFusion
from .calfuse import CalFuseFusion
from .calfuse_conformal import ConformalCalFuse
from .calfuse_copula import CopulaCalFuse
from .linear_learned import LinearLearnedFusion
from .multicalibration import (
    Multicalibration,
    query_length_subgroups,
    signal_dominance_subgroups,
    trivial_subgroup,
    worst_subgroup_ece,
)
from .reranker_fusion import RerankerFusion
from .rrf import RRFFusion

__all__ = [
    "BaseFusion",
    "CalFuseFusion",
    "ConformalCalFuse",
    "CopulaCalFuse",
    "LinearLearnedFusion",
    "Multicalibration",
    "RerankerFusion",
    "RRFFusion",
    "query_length_subgroups",
    "signal_dominance_subgroups",
    "trivial_subgroup",
    "worst_subgroup_ece",
]
