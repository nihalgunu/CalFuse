"""Mondrian-Venn-Abers: signal-stratified distribution-free calibration.

Mondrian conformal prediction (Vovk, 2003; Vovk, Lindsay, Nouretdinov,
Gammerman 2003) partitions the input space into disjoint strata and
runs conformal prediction within each stratum, producing conditionally
valid prediction sets. We apply the same partitioning idea to
Venn-Abers: per-stratum IVAP envelopes are conditionally valid within
each stratum, which is a strictly stronger guarantee than marginal
Venn-Abers.

Key theoretical point (Theorem 5 in ``theory/proofs.tex``): if the
stratification function ``g`` is measurable with respect to the
*signal vector* (it may depend on any function of ``S_1, \\ldots, S_n``
but not on ``Y``), then exchangeability is preserved within each
stratum, so per-stratum Venn-Abers guarantees transfer to the
fused predictor without additional assumptions. This is the part
that is new: we exploit a retrieval-specific stratification
(signal-family dominance) that happens to be measurable in the
signals alone.

Novelty
-------
* Venn-Abers has been applied to single-model scores (Vovk-Petej 2014)
  and to ensembles of identically-typed learners. Its application to
  heterogeneous retrieval-signal fusion is new.
* The signal-measurable-stratification requirement is new as a
  conditional-validity knob for retrieval; it is orthogonal to the
  subgroup-family requirement of multicalibration (Theorem 4) and
  thus the two layers compose.

Usage
-----
``MondrianVennAbers`` takes a base fusion rule and a subgroup function
(the same shape as used by :mod:`src.fusion.multicalibration`). It
fits per-subgroup IVAPs on the calibration split and returns an
envelope ``[p_lo, p_hi]`` per test pair. The envelope width is
reported as a sharpness metric; coverage is measured empirically in
the Phase-3 evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..fusion.base import BaseFusion
from ..fusion.multicalibration import trivial_subgroup
from .venn_abers import FastVennAbersPredictor, VennAbersEnvelope


SubgroupFn = Callable[[np.ndarray, Sequence[str]], np.ndarray]


@dataclass
class MondrianReport:
    per_subgroup_widths: np.ndarray
    per_subgroup_counts: np.ndarray
    per_subgroup_empirical_coverage: Optional[np.ndarray] = None


class MondrianVennAbers(BaseFusion):
    """Signal-stratified Venn-Abers on top of a base fusion rule.

    Parameters
    ----------
    base : BaseFusion
        Any fusion rule producing probability outputs. Will be
        fit inside :meth:`fit`.
    subgroup_fn : SubgroupFn, optional
        Function mapping ``(scores, query_ids)`` to a
        ``(n_pairs, n_groups)`` boolean membership matrix. **Must be
        measurable with respect to signals alone** (no dependence on
        ``Y``) for Theorem 5 to apply. Default is the trivial
        single-group partition, which recovers marginal Venn-Abers.
    fast : bool
        If True, use the amortised PAV approximation
        (:class:`FastVennAbersPredictor`); otherwise use the exact
        IVAP. The default ``True`` is tens of times faster at large
        batch sizes.
    """

    name = "mondrian_venn_abers"

    def __init__(
        self,
        base: BaseFusion,
        subgroup_fn: Optional[SubgroupFn] = None,
        fast: bool = True,
        min_stratum: int = 30,
        min_positives: int = 10,
    ) -> None:
        self.base = base
        self.subgroup_fn = subgroup_fn or trivial_subgroup()
        self.fast = bool(fast)
        self.min_stratum = int(min_stratum)
        # Low-base-rate protection: a stratum with fewer than this many positives
        # gives a degenerate Venn-Abers envelope (pinned at width 0.5). When a
        # stratum fails the positive-count floor we fall back to the pooled
        # global IVAP, which at least has the full positive budget to fit.
        self.min_positives = int(min_positives)
        self._ivaps: dict[int, FastVennAbersPredictor] = {}
        # Fallback IVAP for under-populated strata.
        self._global_ivap: Optional[FastVennAbersPredictor] = None
        self.report_: MondrianReport = MondrianReport(
            per_subgroup_widths=np.zeros(0),
            per_subgroup_counts=np.zeros(0),
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "MondrianVennAbers":
        if labels is None:
            raise ValueError("MondrianVennAbers requires labels to fit IVAPs")

        self.base.fit(scores, labels=labels, query_ids=query_ids)
        p = self.base.fuse(scores, query_ids=query_ids)
        y = np.asarray(labels, dtype=np.int64)
        M = np.asarray(self.subgroup_fn(scores, query_ids or []), dtype=bool)
        if M.ndim == 1:
            M = M[:, None]

        # Global IVAP as a fallback for under-populated strata.
        self._global_ivap = FastVennAbersPredictor().fit(p, y)

        self._ivaps = {}
        widths = []
        counts = []
        for g in range(M.shape[1]):
            mask = M[:, g]
            n = int(mask.sum())
            n_pos = int(y[mask].sum())
            counts.append(n)
            if n < self.min_stratum or n_pos < self.min_positives:
                # Leave this stratum to the global IVAP fallback — per-stratum
                # Venn-Abers on fewer than ``min_positives`` positives degenerates
                # to a width-0.5 envelope that wrecks marginal ECE.
                widths.append(np.nan)
                continue
            ivap = FastVennAbersPredictor().fit(p[mask], y[mask])
            env = ivap.predict_envelope(p[mask])
            widths.append(float(np.mean(env.width)))
            self._ivaps[g] = ivap
        self.report_ = MondrianReport(
            per_subgroup_widths=np.array(widths, dtype=np.float64),
            per_subgroup_counts=np.array(counts, dtype=np.int64),
        )
        return self

    # ------------------------------------------------------------------
    def _envelope(
        self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None
    ) -> VennAbersEnvelope:
        p = self.base.fuse(scores, query_ids=query_ids)
        M = np.asarray(self.subgroup_fn(scores, query_ids or []), dtype=bool)
        if M.ndim == 1:
            M = M[:, None]
        p_lo = np.empty_like(p)
        p_hi = np.empty_like(p)
        # Each row is assigned to its first matching stratum with a
        # fitted IVAP; if no stratum is fit, fall back to the global one.
        assert self._global_ivap is not None
        handled = np.zeros_like(p, dtype=bool)
        for g, ivap in self._ivaps.items():
            mask = M[:, g] & (~handled)
            if not mask.any():
                continue
            env = ivap.predict_envelope(p[mask])
            p_lo[mask] = env.p_lo
            p_hi[mask] = env.p_hi
            handled[mask] = True
        if (~handled).any():
            env = self._global_ivap.predict_envelope(p[~handled])
            p_lo[~handled] = env.p_lo
            p_hi[~handled] = env.p_hi
        return VennAbersEnvelope(p_lo=p_lo, p_hi=p_hi)

    def predict_envelope(
        self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None
    ) -> VennAbersEnvelope:
        return self._envelope(scores, query_ids)

    def fuse(
        self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        return self._envelope(scores, query_ids).midpoint
