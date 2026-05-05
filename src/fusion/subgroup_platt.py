"""Subgroup-stratified Platt wrapper.

A simpler alternative to Multi-CalFuse's additive-logit-per-cell
corrections: fit one Platt calibrator per subgroup on the base
fusion's output, then route each test pair to its subgroup's
calibrator.

This is the natural ``per-subgroup calibration'' baseline a
reviewer asks about: does Multi-CalFuse's HKRR-style additive
correction matter, or does a per-subgroup Platt suffice?

Algorithmically:
    For each subgroup g in G:
        Fit Platt_g on { (base.fuse(X_j), Y_j) : g(X_j) = g }
    At inference:
        For each test pair X, find its subgroup g(X), apply Platt_g.

If a stratum is too small (< n_min positives), fall back to a
global Platt fit on the full calibration set. This matches the
Mondrian-Venn-Abers fallback policy.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ..calibrators.isotonic import IsotonicCalibrator
from ..calibrators.platt import PlattCalibrator
from .base import BaseFusion
from .multicalibration import signal_dominance_subgroups


SubgroupFn = Callable[[np.ndarray, Sequence[str]], np.ndarray]


class SubgroupStratifiedCalibrator(BaseFusion):
    """Generic per-subgroup calibrator wrapper.

    Subclasses fix the per-subgroup calibrator (Platt, isotonic).
    Same fallback policy as :class:`SubgroupStratifiedPlatt`.
    """

    name = "subgroup_stratified_calibrator"
    _calibrator_factory = PlattCalibrator  # subclasses override

    def __init__(
        self,
        base: BaseFusion,
        subgroup_fn: Optional[SubgroupFn] = None,
        n_min: int = 30,
        n_pos_min: int = 10,
    ) -> None:
        self.base = base
        self.subgroup_fn = subgroup_fn or signal_dominance_subgroups()
        self.n_min = int(n_min)
        self.n_pos_min = int(n_pos_min)
        self._per_subgroup = {}
        self._global = None

    def fit(self, scores, labels=None, query_ids=None):
        if labels is None:
            raise ValueError("requires labels")
        self.base.fit(scores, labels=labels, query_ids=query_ids)
        p_base = self.base.fuse(scores, query_ids=query_ids)
        y = np.asarray(labels, dtype=np.int64)
        M = np.asarray(self.subgroup_fn(scores, query_ids or []), dtype=bool)
        if M.ndim == 1:
            M = M[:, None]
        self._global = self._calibrator_factory()
        self._global.fit(p_base, y)
        self._per_subgroup = {}
        for g in range(M.shape[1]):
            mask = M[:, g]
            if mask.sum() < self.n_min or y[mask].sum() < self.n_pos_min:
                continue
            cal = self._calibrator_factory()
            cal.fit(p_base[mask], y[mask])
            self._per_subgroup[g] = cal
        return self

    def fuse(self, scores, query_ids=None):
        p_base = self.base.fuse(scores, query_ids=query_ids).astype(np.float64)
        M = np.asarray(self.subgroup_fn(scores, query_ids or []), dtype=bool)
        if M.ndim == 1:
            M = M[:, None]
        out = np.empty_like(p_base)
        handled = np.zeros_like(p_base, dtype=bool)
        for g, cal in self._per_subgroup.items():
            mask = M[:, g] & (~handled)
            if not mask.any():
                continue
            out[mask] = cal.transform(p_base[mask])
            handled[mask] = True
        if (~handled).any():
            assert self._global is not None
            out[~handled] = self._global.transform(p_base[~handled])
        return out


class SubgroupStratifiedIsotonic(SubgroupStratifiedCalibrator):
    """Per-subgroup isotonic calibrator wrapper. Non-parametric
    analog of :class:`SubgroupStratifiedPlatt`."""
    name = "subgroup_stratified_isotonic"
    _calibrator_factory = IsotonicCalibrator


class SubgroupStratifiedPlatt(BaseFusion):
    """Per-subgroup Platt calibrator wrapper for any base fusion.

    Parameters
    ----------
    base
        Base fusion rule producing a probability output. Will be
        fit inside ``fit``.
    subgroup_fn
        Function returning ``(n_pairs, n_groups)`` boolean
        membership matrix. Default: signal-family dominance.
    n_min
        Minimum stratum size; under-populated strata fall back to
        the global Platt.
    n_pos_min
        Minimum positives per stratum; under-supported strata fall
        back to the global Platt.
    """

    name = "subgroup_stratified_platt"

    def __init__(
        self,
        base: BaseFusion,
        subgroup_fn: Optional[SubgroupFn] = None,
        n_min: int = 30,
        n_pos_min: int = 10,
    ) -> None:
        self.base = base
        self.subgroup_fn = subgroup_fn or signal_dominance_subgroups()
        self.n_min = int(n_min)
        self.n_pos_min = int(n_pos_min)
        self._per_subgroup: dict[int, PlattCalibrator] = {}
        self._global: Optional[PlattCalibrator] = None

    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "SubgroupStratifiedPlatt":
        if labels is None:
            raise ValueError("SubgroupStratifiedPlatt requires labels")
        self.base.fit(scores, labels=labels, query_ids=query_ids)
        p_base = self.base.fuse(scores, query_ids=query_ids)
        y = np.asarray(labels, dtype=np.int64)
        M = np.asarray(self.subgroup_fn(scores, query_ids or []), dtype=bool)
        if M.ndim == 1:
            M = M[:, None]

        # Global fallback Platt on full calibration set.
        self._global = PlattCalibrator()
        self._global.fit(p_base, y)

        self._per_subgroup = {}
        for g in range(M.shape[1]):
            mask = M[:, g]
            if mask.sum() < self.n_min:
                continue
            if y[mask].sum() < self.n_pos_min:
                continue
            cal = PlattCalibrator()
            cal.fit(p_base[mask], y[mask])
            self._per_subgroup[g] = cal
        return self

    def fuse(
        self,
        scores: np.ndarray,
        query_ids: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        p_base = self.base.fuse(scores, query_ids=query_ids).astype(np.float64)
        M = np.asarray(self.subgroup_fn(scores, query_ids or []), dtype=bool)
        if M.ndim == 1:
            M = M[:, None]
        out = np.empty_like(p_base)
        # First-match assignment: for each pair, use the first subgroup
        # it belongs to that has a fitted calibrator; else fall back
        # to global Platt.
        handled = np.zeros_like(p_base, dtype=bool)
        for g, cal in self._per_subgroup.items():
            mask = M[:, g] & (~handled)
            if not mask.any():
                continue
            out[mask] = cal.transform(p_base[mask])
            handled[mask] = True
        if (~handled).any():
            assert self._global is not None
            out[~handled] = self._global.transform(p_base[~handled])
        return out
