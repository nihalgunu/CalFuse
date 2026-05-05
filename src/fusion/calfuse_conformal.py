"""Conformal-CalFuse: distribution-free wrapper with Venn-Abers envelopes.

Wraps any :class:`BaseFusion` with a :class:`MondrianVennAbers` layer,
so the wrapper exposes three outputs per test pair:

* a point probability ``p(x)`` (envelope midpoint, used by code paths
  that expect :meth:`BaseFusion.fuse` to return a scalar);
* the envelope ``[p_lo(x), p_hi(x)]`` with distribution-free
  finite-sample calibration guarantees under exchangeability of
  calibration / test splits (Theorem 5);
* the envelope width, a sharpness metric complementary to ECE.

For the anytime-valid sequential-fusion decision rule (Theorem 6), see
:mod:`src.conformal.sequential`.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .base import BaseFusion
from .calfuse import CalFuseFusion

if False:  # pragma: no cover - type-checking only
    from ..conformal.venn_abers import VennAbersEnvelope  # noqa: F401


SubgroupFn = Callable[[np.ndarray, Sequence[str]], np.ndarray]


class ConformalCalFuse(BaseFusion):
    name = "calfuse_conformal"

    def __init__(
        self,
        base: Optional[BaseFusion] = None,
        subgroup_fn: Optional[SubgroupFn] = None,
        fast: bool = True,
    ) -> None:
        # Lazy import to avoid a circular dependency between the
        # ``fusion`` and ``conformal`` packages (``conformal.mondrian``
        # imports ``fusion.base``).
        from ..conformal.mondrian import MondrianVennAbers

        if base is None:
            base = CalFuseFusion(force_mode="parametric")
        self._mva = MondrianVennAbers(base=base, subgroup_fn=subgroup_fn, fast=fast)

    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "ConformalCalFuse":
        self._mva.fit(scores, labels=labels, query_ids=query_ids)
        return self

    def fuse(
        self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        return self._mva.fuse(scores, query_ids=query_ids)

    def predict_envelope(
        self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None
    ):
        return self._mva.predict_envelope(scores, query_ids=query_ids)
