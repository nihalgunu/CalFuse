"""Single evaluation entry point.

Metrics
-------
* **ECE** — Expected Calibration Error with equal-width binning, 15 bins
  (Naeini, Cooper, Hauskrecht, 2015; Guo et al., 2017). Equal-width is
  preferred over equal-frequency because reliability-diagram columns
  need to line up across methods for visual comparison. The benchmark
  reports both ``n_bins=15`` and a robustness check at ``n_bins=10``.
* **MCE** — Maximum Calibration Error; the worst-bin deviation.
* **Brier** — mean squared error between probability and label (Brier,
  1950); decomposes into reliability + resolution - uncertainty
  (Murphy, 1973).
* **NLL** — negative log-likelihood; proper scoring rule.
* **NDCG@k** — normalised discounted cumulative gain, graded-relevance
  formulation (Järvelin & Kekäläinen, 2002).
* **Selective accuracy / coverage curve** — downstream-decision metric
  used in Phase 3 abstention experiments (El-Yaniv & Wiener, 2010).

All functions take NumPy inputs and return either scalars or simple
dataclasses; there is no machine-state dependency. Reliability-diagram
bar data is returned even when matplotlib is unavailable so the
benchmark can be analysed offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
@dataclass
class ReliabilityDiagram:
    bin_edges: np.ndarray
    bin_counts: np.ndarray
    bin_confidences: np.ndarray
    bin_accuracies: np.ndarray

    def as_dict(self) -> dict:
        return {
            "bin_edges": self.bin_edges.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "bin_confidences": self.bin_confidences.tolist(),
            "bin_accuracies": self.bin_accuracies.tolist(),
        }


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> ReliabilityDiagram:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    conf = np.zeros(n_bins, dtype=np.float64)
    acc = np.zeros(n_bins, dtype=np.float64)

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (probs >= lo) & ((probs < hi) if b < n_bins - 1 else (probs <= hi))
        n = int(mask.sum())
        counts[b] = n
        if n > 0:
            conf[b] = float(probs[mask].mean())
            acc[b] = float(labels[mask].mean())

    return ReliabilityDiagram(edges, counts, conf, acc)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    diag = reliability_diagram(probs, labels, n_bins)
    total = diag.bin_counts.sum()
    if total == 0:
        return 0.0
    weights = diag.bin_counts / total
    return float(np.sum(weights * np.abs(diag.bin_accuracies - diag.bin_confidences)))


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    diag = reliability_diagram(probs, labels, n_bins)
    mask = diag.bin_counts > 0
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(diag.bin_accuracies[mask] - diag.bin_confidences[mask])))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    return float(np.mean((probs - labels) ** 2))


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), 1e-12, 1 - 1e-12)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def ndcg_at_k(
    scores: np.ndarray,
    graded_labels: np.ndarray,
    query_ids: Sequence[str],
    k: int = 10,
) -> float:
    """Graded-relevance NDCG@k averaged over queries (Järvelin & Kekäläinen, 2002)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(graded_labels, dtype=np.float64).reshape(-1)

    by_q: dict[str, list[int]] = {}
    for i, qid in enumerate(query_ids):
        by_q.setdefault(qid, []).append(i)

    gains = []
    for idxs in by_q.values():
        idxs = np.array(idxs, dtype=np.int64)
        s = scores[idxs]
        y = labels[idxs]
        order = np.argsort(-s, kind="mergesort")[:k]
        y_ranked = y[order]
        discounts = 1.0 / np.log2(np.arange(2, len(y_ranked) + 2))
        dcg = float(np.sum((2 ** y_ranked - 1) * discounts))
        ideal_order = np.argsort(-y, kind="mergesort")[:k]
        y_ideal = y[ideal_order]
        discounts_i = 1.0 / np.log2(np.arange(2, len(y_ideal) + 2))
        idcg = float(np.sum((2 ** y_ideal - 1) * discounts_i))
        if idcg > 0:
            gains.append(dcg / idcg)
    return float(np.mean(gains)) if gains else 0.0


# ---------------------------------------------------------------------------
# Selective-accuracy curve (downstream-decision metric)
# ---------------------------------------------------------------------------
@dataclass
class SelectivePoint:
    coverage: float
    selective_accuracy: float


@dataclass
class SelectiveCurve:
    points: list[SelectivePoint] = field(default_factory=list)

    def at_coverage(self, target: float) -> float:
        """Selective accuracy at the *smallest* coverage >= target."""
        for p in self.points:
            if p.coverage >= target:
                return p.selective_accuracy
        return float("nan")


def selective_accuracy_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> SelectiveCurve:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    # Decision rule: predict positive iff probability >= threshold;
    # abstain iff probability is within an uncertainty band ``[lo, hi]``
    # around 0.5. For simplicity here we use a one-sided rule: accept
    # prediction iff ``max(p, 1-p) >= threshold``; ``coverage`` is the
    # fraction of pairs where the rule accepts; ``selective accuracy``
    # is the 0-1 accuracy on the accepted subset.
    pts = []
    for t in thresholds:
        conf = np.maximum(probs, 1 - probs)
        accept = conf >= t
        cov = float(accept.mean())
        if cov == 0:
            continue
        preds = (probs >= 0.5).astype(np.int64)
        acc = float((preds[accept] == labels[accept]).mean())
        pts.append(SelectivePoint(coverage=cov, selective_accuracy=acc))
    # Sort by coverage ascending, then de-duplicate.
    pts.sort(key=lambda p: p.coverage)
    return SelectiveCurve(points=pts)


# ---------------------------------------------------------------------------
# Batch evaluator
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    ece_15: float
    ece_10: float
    mce_15: float
    brier: float
    nll: float
    ndcg_10: Optional[float] = None
    reliability: Optional[ReliabilityDiagram] = None

    def as_dict(self) -> dict:
        out = {
            "ece_15": self.ece_15,
            "ece_10": self.ece_10,
            "mce_15": self.mce_15,
            "brier": self.brier,
            "nll": self.nll,
            "ndcg_10": self.ndcg_10,
        }
        if self.reliability is not None:
            out["reliability"] = self.reliability.as_dict()
        return out


def evaluate(
    probs: np.ndarray,
    labels: np.ndarray,
    graded_labels: Optional[np.ndarray] = None,
    query_ids: Optional[Sequence[str]] = None,
    include_reliability: bool = True,
) -> EvaluationResult:
    res = EvaluationResult(
        ece_15=expected_calibration_error(probs, labels, 15),
        ece_10=expected_calibration_error(probs, labels, 10),
        mce_15=maximum_calibration_error(probs, labels, 15),
        brier=brier_score(probs, labels),
        nll=negative_log_likelihood(probs, labels),
    )
    if graded_labels is not None and query_ids is not None:
        res.ndcg_10 = ndcg_at_k(probs, graded_labels, query_ids, k=10)
    if include_reliability:
        res.reliability = reliability_diagram(probs, labels, 15)
    return res
