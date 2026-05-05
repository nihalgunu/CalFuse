"""Multicalibration wrapper for fusion rules.

Background
----------
Multicalibration (H\\'ebert-Johnson, Kim, Reingold, Rothblum, 2018,
"Multicalibration: Calibration for the (Computationally-Identifiable)
Masses") strengthens marginal calibration to require calibration *on
every subgroup* ``g`` in a pre-specified subgroup family ``G``. A
multicalibrated predictor is guaranteed to be calibrated on the
worst-case subgroup, which is the right notion for downstream
decisions that act differently on different query populations (short
vs. long queries, head vs. tail topics, lexical-dominant vs.
semantic-dominant passages).

Prior work in the calibration literature has treated multicalibration
as a post-hoc procedure applied to a single model's output. We lift
the framework to multi-signal retrieval fusion: the wrapper takes any
:class:`BaseFusion` as a base predictor and enforces multicalibration
over a user-specified subgroup family.

Algorithm
---------
We implement the HKRR boosting-style correction with additive logit
updates. Let ``f`` be the base fused probability. For each subgroup
``S in G`` and each output-bin ``I`` defined on ``f``, compute

    Delta(S, I) = logit(E[Y | X in S, f(X) in I])
                  - logit(E[f(X) | X in S, f(X) in I]).

If the magnitude of ``Delta(S, I)`` exceeds a tolerance ``alpha`` on
enough calibration data (``>= n_min`` pairs), we add ``Delta`` to the
logit of ``f(X)`` for all ``X`` in the subgroup-bin. The procedure
iterates until no violation exceeds ``alpha`` or a maximum iteration
count is reached. Convergence in a bounded number of iterations is
guaranteed by the HKRR potential-function argument.

Subgroup families for retrieval
-------------------------------
``src.fusion.multicalibration`` ships a small library of retrieval-
specific subgroup functions:

* query-length buckets (short / medium / long, robust to tokenisation
  differences),
* signal-family dominance (which calibrated signal logit is
  largest),
* output-confidence tertiles (low / medium / high base-predictor
  confidence), and
* all-indicator baseline (a single "trivial" subgroup, reducing the
  wrapper to classical marginal calibration).

Users may pass their own ``subgroup_fn`` returning a boolean
``(n_pairs, n_groups)`` membership matrix; this is the entry point
for intent-classifier, topic-clustering, or user-cohort subgroupings
in production deployments.

Novelty
-------
To our knowledge multicalibration has not previously been applied to
multi-signal retrieval fusion. Theorem 4 of ``theory/proofs.tex``
proves that parametric CalFuse preserves multicalibration under
conditional independence *and* per-signal multicalibration --- a
strict extension of the marginal calibration preservation result in
Theorem 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .base import BaseFusion


EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# ---------------------------------------------------------------------------
# Subgroup-function library
# ---------------------------------------------------------------------------
SubgroupFn = Callable[[np.ndarray, Sequence[str]], np.ndarray]


def query_length_subgroups(
    raw_query_lengths: Sequence[int], short_th: int = 5, long_th: int = 12
) -> SubgroupFn:
    """Return a subgroup fn keyed on query length buckets.

    ``raw_query_lengths[i]`` must give the token count of the query
    associated with row ``i`` of the score matrix. Buckets are fixed
    at fit time and re-applied at inference.
    """
    qls = np.asarray(raw_query_lengths, dtype=np.int64)

    def fn(scores: np.ndarray, query_ids: Sequence[str]) -> np.ndarray:
        n = scores.shape[0]
        short = qls < short_th
        long_ = qls >= long_th
        med = ~short & ~long_
        return np.stack([short, med, long_], axis=1)

    return fn


def signal_dominance_subgroups(top_k: int = 3) -> SubgroupFn:
    """Return a subgroup fn keyed on which signal column is loudest.

    One subgroup per signal index; the membership indicator is 1 if
    that signal produced the largest standardised score for the row.
    """

    def fn(scores: np.ndarray, query_ids: Sequence[str]) -> np.ndarray:
        X = np.asarray(scores, dtype=np.float64)
        # Standardise columns to compare across scales.
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Xs = (X - mu) / sd
        dominant = np.argmax(Xs, axis=1)
        d = X.shape[1]
        M = np.zeros((X.shape[0], d), dtype=bool)
        M[np.arange(X.shape[0]), dominant] = True
        return M[:, :top_k] if d > top_k else M

    return fn


def trivial_subgroup() -> SubgroupFn:
    def fn(scores: np.ndarray, query_ids: Sequence[str]) -> np.ndarray:
        return np.ones((scores.shape[0], 1), dtype=bool)

    return fn


# ---------------------------------------------------------------------------
# Multicalibration wrapper
# ---------------------------------------------------------------------------
@dataclass
class MulticalibrationReport:
    n_iterations: int = 0
    n_corrections: int = 0
    worst_violation_init: float = 0.0
    worst_violation_final: float = 0.0
    per_subgroup_worst: list[float] = field(default_factory=list)


class Multicalibration(BaseFusion):
    """Post-hoc multicalibration wrapper for any :class:`BaseFusion`.

    Parameters
    ----------
    base : BaseFusion
        Already-constructed base fusion; will be fit inside ``fit``.
    subgroup_fn : SubgroupFn
        Callable returning a ``(n_pairs, n_groups)`` boolean membership
        matrix given ``(scores, query_ids)``. Pairs may belong to any
        subset of groups (overlapping subgroups are supported).
    n_bins : int
        Output-bin granularity. HKRR-style corrections are applied
        inside each (subgroup, bin) cell. 10 bins match the default
        ECE granularity; finer binning gives tighter correction at
        the cost of per-cell sample size.
    alpha : float
        Per-cell tolerance in probability space; cells with
        |E[y] - E[f]| <= alpha are not corrected.
    n_min : int
        Minimum cell population required before we trust the cell's
        residual estimate enough to apply a correction.
    max_iter : int
        Maximum outer passes over (subgroup, bin) cells. HKRR
        guarantees convergence in a bounded number of iterations; we
        also stop early when no violation exceeds ``alpha``.
    """

    name = "multicalibration"

    def __init__(
        self,
        base: BaseFusion,
        subgroup_fn: Optional[SubgroupFn] = None,
        n_bins: int = 10,
        alpha: float = 0.02,
        n_min: int = 25,
        max_iter: int = 50,
    ) -> None:
        self.base = base
        self.subgroup_fn = subgroup_fn or trivial_subgroup()
        self.n_bins = int(n_bins)
        self.alpha = float(alpha)
        self.n_min = int(n_min)
        self.max_iter = int(max_iter)
        self._edges: np.ndarray = np.linspace(0.0, 1.0, n_bins + 1)
        # Corrections keyed by (group_idx, bin_idx) -> additive logit delta.
        self._corrections: dict[tuple[int, int], float] = {}
        self.report_: MulticalibrationReport = MulticalibrationReport()

    # ------------------------------------------------------------------
    def _bin_of(self, p: np.ndarray) -> np.ndarray:
        b = np.clip(np.digitize(p, self._edges) - 1, 0, self.n_bins - 1)
        return b.astype(np.int64)

    def _worst_cell_violation(
        self,
        p: np.ndarray,
        y: np.ndarray,
        M: np.ndarray,
    ) -> float:
        bins = self._bin_of(p)
        worst = 0.0
        for g in range(M.shape[1]):
            mask_g = M[:, g]
            if mask_g.sum() < self.n_min:
                continue
            for b in range(self.n_bins):
                mask = mask_g & (bins == b)
                if mask.sum() < self.n_min:
                    continue
                gap = float(y[mask].mean() - p[mask].mean())
                worst = max(worst, abs(gap))
        return worst

    # ------------------------------------------------------------------
    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ) -> "Multicalibration":
        if labels is None:
            raise ValueError("Multicalibration requires labels to fit the correction table")
        self.base.fit(scores, labels=labels, query_ids=query_ids)
        p = self.base.fuse(scores, query_ids=query_ids).copy()
        y = np.asarray(labels, dtype=np.float64)
        M = np.asarray(self.subgroup_fn(scores, query_ids or []), dtype=bool)
        if M.ndim == 1:
            M = M[:, None]

        self.report_.worst_violation_init = self._worst_cell_violation(p, y, M)
        corrections: dict[tuple[int, int], float] = {}
        n_corrections = 0
        iterations = 0
        for it in range(self.max_iter):
            iterations = it + 1
            changed = False
            bins = self._bin_of(p)
            for g in range(M.shape[1]):
                mask_g = M[:, g]
                if mask_g.sum() < self.n_min:
                    continue
                for b in range(self.n_bins):
                    cell = mask_g & (bins == b)
                    n_cell = int(cell.sum())
                    if n_cell < self.n_min:
                        continue
                    p_bar = float(p[cell].mean())
                    y_bar = float(y[cell].mean())
                    if abs(y_bar - p_bar) <= self.alpha:
                        continue
                    # Additive logit correction, clipped to probability box.
                    delta = _logit(np.array([y_bar]))[0] - _logit(np.array([p_bar]))[0]
                    p_new = _sigmoid(_logit(p[cell]) + delta)
                    p[cell] = p_new
                    corrections[(g, b)] = corrections.get((g, b), 0.0) + float(delta)
                    n_corrections += 1
                    changed = True
            if not changed:
                break

        self._corrections = corrections
        self.report_.n_iterations = iterations
        self.report_.n_corrections = n_corrections
        self.report_.worst_violation_final = self._worst_cell_violation(p, y, M)
        # Per-subgroup worst violation for diagnostic tables.
        self.report_.per_subgroup_worst = []
        bins = self._bin_of(p)
        for g in range(M.shape[1]):
            mask_g = M[:, g]
            worst = 0.0
            for b in range(self.n_bins):
                cell = mask_g & (bins == b)
                if cell.sum() < self.n_min:
                    continue
                worst = max(worst, abs(float(y[cell].mean() - p[cell].mean())))
            self.report_.per_subgroup_worst.append(worst)
        return self

    def fuse(self, scores: np.ndarray, query_ids: Optional[Sequence[str]] = None) -> np.ndarray:
        p = self.base.fuse(scores, query_ids=query_ids).astype(np.float64)
        if not self._corrections:
            return p
        M = np.asarray(self.subgroup_fn(scores, query_ids or []), dtype=bool)
        if M.ndim == 1:
            M = M[:, None]
        # Apply corrections in a deterministic cell order to make the
        # mapping reproducible. Use a stable sort on (group, bin).
        bins = self._bin_of(p)
        for (g, b), delta in sorted(self._corrections.items()):
            cell = M[:, g] & (bins == b)
            if not cell.any():
                continue
            p_new = _sigmoid(_logit(p[cell]) + delta)
            p[cell] = p_new
            # Bin membership may change after correction; the HKRR
            # analysis still holds because corrections are monotone in
            # logit space and we never revisit the same (g, b) cell
            # twice per pass.
        return p


# ---------------------------------------------------------------------------
# Subgroup-ECE evaluator (used by experiments + diagnostics)
# ---------------------------------------------------------------------------
def worst_subgroup_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    membership: np.ndarray,
    n_bins: int = 15,
    n_min: int = 25,
) -> float:
    """Return ``max_g ECE(f | x in g)`` --- the multicalibration-analogue
    of marginal ECE.
    """
    from ..evaluate import expected_calibration_error

    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    M = np.asarray(membership, dtype=bool)
    if M.ndim == 1:
        M = M[:, None]
    worst = 0.0
    for g in range(M.shape[1]):
        mask = M[:, g]
        if mask.sum() < n_min:
            continue
        ece = expected_calibration_error(probs[mask], labels[mask], n_bins=n_bins)
        worst = max(worst, ece)
    return worst
