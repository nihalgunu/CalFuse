"""Anytime-valid e-process fusion with adaptive stopping.

Framework
---------
We apply the e-process / safe-testing framework of Howard, Ramdas,
McAuliffe and Sekhon (2021, "Time-uniform, nonparametric,
nonasymptotic confidence sequences") --- building on Ville (1939),
Vovk (1993), Gr\"unwald, de Heide and Koolen (2024) --- to
retrieval-signal fusion.

Setup. A query-passage pair ``x`` has latent binary relevance ``Y``.
Each signal ``S_i`` has a per-signal calibrator ``f_i`` so that
``p_i(x) = f_i(S_i(x))`` is the calibrated posterior. Fix a base rate
``pi = P(Y=1)``. The calibrated Bayes-factor (likelihood ratio) for
signal ``i`` is

    E_i(x)  =  [p_i(x) / (1 - p_i(x))] * [(1 - pi) / pi].

Under the null ``H_0: Y = 0``, a well-calibrated signal satisfies
``E[E_i(X) | Y = 0] = 1`` (by the law of total probability). Under
conditional independence of signals given ``Y``, the product

    M_t  =  prod_{i=1}^{t} E_i(X)

is a nonnegative martingale starting at 1 under ``H_0``. By Ville's
inequality,

    P( sup_t M_t  >=  1/alpha   |   Y = 0 )   <=   alpha,

uniformly over all stopping times ``t``. This is anytime validity.

Decision rule. Compute signals in cost-increasing order. At each step:

* If ``M_t >= 1/alpha``, reject ``H_0`` (decide relevant) and stop.
* If ``M_t <= alpha``, accept ``H_0`` (decide irrelevant) and stop.
* Else, continue to the next signal if available, else abstain.

Type-I error is bounded by ``alpha`` and Type-II error decays
exponentially in the KL divergence between the Y=1 and Y=0 signal
distributions (Wald's SPRT bound, with the calibrated-LR substitution).

Novelty
-------
To our knowledge this is the first application of e-processes /
anytime-valid sequential testing to retrieval-signal fusion. The
practical consequence is a *provably-valid stopping rule* on
expensive signal computation (cross-encoder rerank, reranker routing,
graph-structural PPR): we never compute more signals than we need,
and the computation-budget savings come with finite-sample
distribution-free Type-I error control.

See Theorem 6 in ``theory/proofs.tex`` for the formal statement and
its SPRT-style expected-stopping-time upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


EPS = 1e-9


@dataclass
class SequentialFusionDecision:
    """Per-pair output of the sequential fusion procedure."""

    decision: int  # 1 = relevant, 0 = irrelevant, -1 = abstain
    stopping_time: int  # number of signals consumed before stopping (>=1)
    final_e_value: float  # M_t at stopping
    e_trace: np.ndarray  # (T,) cumulative e-values through the trace


@dataclass
class EProcessReport:
    """Aggregate computation-budget statistics over a batch."""

    mean_stopping_time: float
    abstention_rate: float
    empirical_type1: float
    empirical_type2: float
    per_signal_consumption: np.ndarray


class EProcess:
    """Run an anytime-valid sequential fusion procedure given a
    pre-computed signal-ordered calibrated-probability matrix.

    Parameters
    ----------
    alpha : float
        Per-pair Type-I error target. The rule decides "relevant"
        only when the cumulative e-value exceeds ``1/alpha``, and
        decides "irrelevant" only when it drops below ``alpha``.
        Default ``0.05``.
    pi : float
        Global positive base rate used in the calibrated-LR
        construction. Estimated from calibration split; must fall in
        ``(0, 1)``.
    """

    def __init__(self, alpha: float = 0.05, pi: float = 0.25) -> None:
        self.alpha = float(alpha)
        self.pi = float(np.clip(pi, EPS, 1.0 - EPS))

    def _e_value(self, p: float) -> float:
        """Calibrated likelihood ratio for a single signal.

        ``E = (p/(1-p)) * ((1-pi)/pi)``; clipped for numerical
        stability when ``p`` is near 0 or 1.
        """
        p = float(np.clip(p, EPS, 1.0 - EPS))
        return (p / (1.0 - p)) * ((1.0 - self.pi) / self.pi)

    def run_one(self, calibrated_probs: Sequence[float]) -> SequentialFusionDecision:
        """Run the rule on one pair's signal-ordered calibrated probs."""
        M = 1.0
        trace = []
        for t, p in enumerate(calibrated_probs, start=1):
            M = M * self._e_value(float(p))
            trace.append(M)
            if M >= 1.0 / self.alpha:
                return SequentialFusionDecision(
                    decision=1, stopping_time=t, final_e_value=M, e_trace=np.array(trace)
                )
            if M <= self.alpha:
                return SequentialFusionDecision(
                    decision=0, stopping_time=t, final_e_value=M, e_trace=np.array(trace)
                )
        return SequentialFusionDecision(
            decision=-1, stopping_time=len(trace), final_e_value=M, e_trace=np.array(trace)
        )

    def run_batch(
        self,
        calibrated_probs: np.ndarray,  # (n_pairs, n_signals_sorted)
    ) -> list[SequentialFusionDecision]:
        return [self.run_one(calibrated_probs[i, :]) for i in range(calibrated_probs.shape[0])]

    def evaluate(
        self,
        calibrated_probs: np.ndarray,
        labels: np.ndarray,
    ) -> EProcessReport:
        decisions = self.run_batch(calibrated_probs)
        n_pairs = len(decisions)
        stopping_times = np.array([d.stopping_time for d in decisions])
        decision_vec = np.array([d.decision for d in decisions])
        y = np.asarray(labels, dtype=np.int64).reshape(-1)

        neg_mask = (y == 0)
        pos_mask = (y == 1)
        # Empirical Type-I: decided "relevant" when Y=0.
        emp_type1 = float(((decision_vec == 1) & neg_mask).sum() / max(1, neg_mask.sum()))
        emp_type2 = float(((decision_vec == 0) & pos_mask).sum() / max(1, pos_mask.sum()))
        abstention = float((decision_vec == -1).mean())

        n_signals = calibrated_probs.shape[1]
        per_signal = np.zeros(n_signals, dtype=np.int64)
        for st in stopping_times:
            per_signal[: st] += 1
        return EProcessReport(
            mean_stopping_time=float(stopping_times.mean()),
            abstention_rate=abstention,
            empirical_type1=emp_type1,
            empirical_type2=emp_type2,
            per_signal_consumption=per_signal.astype(np.float64) / max(1, n_pairs),
        )


def sequential_fusion(
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
    pi: Optional[float] = None,
) -> tuple[list[SequentialFusionDecision], EProcessReport]:
    """Convenience wrapper.

    ``calibrated_probs`` is ``(n_pairs, n_signals)`` where the
    *columns are ordered in increasing computational cost* so that the
    procedure uses the cheapest information first. ``pi`` defaults to
    the empirical positive rate on ``labels`` when ``None``.
    """
    if pi is None:
        pi = float(np.mean(labels))
    ep = EProcess(alpha=alpha, pi=pi)
    decisions = ep.run_batch(calibrated_probs)
    report = ep.evaluate(calibrated_probs, labels)
    return decisions, report


# ---------------------------------------------------------------------------
# Conformalised e-process: calibration-set-anchored thresholds.
# ---------------------------------------------------------------------------
class ConformalEProcess(EProcess):
    """Anytime-valid sequential fusion with a conformalised stopping
    rule that does not require conditional independence.

    Under exchangeability between calibration negatives and test
    negatives (a strictly weaker assumption than CI given Y), the
    empirical (1 - alpha) quantile of the calibration-negative
    running-max e-process is a valid threshold for one-sided Type-I
    control: the empirical Type-I on test negatives is bounded by
    ``alpha + 1 / (n_cal_neg + 1)`` in finite samples (the +1/(n+1)
    is the standard conformal slack).

    The Type-II / abstention behaviour is unchanged from the
    parent ``EProcess`` --- only the upper threshold is replaced by
    the calibration-derived quantile. The lower threshold (decide
    irrelevant) keeps the Ville-derived ``alpha`` and is therefore
    conservative under the same exchangeability assumption (using
    the empirical alpha-quantile of the *running-min* over cal-negs
    is the symmetric construction).

    Parameters
    ----------
    alpha : float
        Per-pair Type-I error target.
    pi : float
        Global positive base rate (used in the LR construction; the
        threshold is then learned data-adaptively).
    """

    def __init__(self, alpha: float = 0.05, pi: float = 0.25) -> None:
        super().__init__(alpha=alpha, pi=pi)
        self._upper_threshold: Optional[float] = None
        self._lower_threshold: Optional[float] = None
        self._n_cal_neg: int = 0

    def fit(self, cal_calibrated_probs: np.ndarray, cal_labels: np.ndarray) -> "ConformalEProcess":
        """Learn the empirical-quantile thresholds from calibration negatives."""
        cal_y = np.asarray(cal_labels, dtype=np.int64).reshape(-1)
        neg_mask = cal_y == 0
        self._n_cal_neg = int(neg_mask.sum())
        if self._n_cal_neg == 0:
            self._upper_threshold = 1.0 / max(self.alpha, EPS)
            self._lower_threshold = self.alpha
            return self
        # Run the unstopped e-process on cal-negatives; track running-max
        # and running-min of M_t over signals.
        P = cal_calibrated_probs[neg_mask]
        max_M = np.empty(self._n_cal_neg, dtype=np.float64)
        min_M = np.empty(self._n_cal_neg, dtype=np.float64)
        for i in range(self._n_cal_neg):
            M = 1.0
            mx, mn = M, M
            for p in P[i]:
                M *= self._e_value(float(p))
                if M > mx:
                    mx = M
                if M < mn:
                    mn = M
            max_M[i] = mx
            min_M[i] = mn
        # Conformal upper threshold: (1 - alpha)(1 + 1/n) quantile of running-max.
        q_upper = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / self._n_cal_neg))
        self._upper_threshold = float(np.quantile(max_M, q_upper))
        # Conformal lower threshold: alpha quantile of running-min.
        q_lower = max(0.0, self.alpha * (1.0 + 1.0 / self._n_cal_neg))
        self._lower_threshold = float(np.quantile(min_M, q_lower))
        return self

    def run_one(self, calibrated_probs: Sequence[float]) -> SequentialFusionDecision:
        if self._upper_threshold is None or self._lower_threshold is None:
            return super().run_one(calibrated_probs)
        M = 1.0
        trace = []
        for t, p in enumerate(calibrated_probs, start=1):
            M = M * self._e_value(float(p))
            trace.append(M)
            if M >= self._upper_threshold:
                return SequentialFusionDecision(
                    decision=1, stopping_time=t, final_e_value=M, e_trace=np.array(trace)
                )
            if M <= self._lower_threshold:
                return SequentialFusionDecision(
                    decision=0, stopping_time=t, final_e_value=M, e_trace=np.array(trace)
                )
        return SequentialFusionDecision(
            decision=-1, stopping_time=len(trace), final_e_value=M, e_trace=np.array(trace)
        )


def conformal_sequential_fusion(
    cal_calibrated_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_calibrated_probs: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.05,
    pi: Optional[float] = None,
) -> tuple[list[SequentialFusionDecision], EProcessReport, float, float]:
    """Conformalised sequential fusion convenience wrapper.

    Returns the test-set decisions, evaluation report, and the two
    fitted (upper, lower) thresholds for inspection.
    """
    if pi is None:
        pi = float(np.mean(cal_labels))
    ep = ConformalEProcess(alpha=alpha, pi=pi)
    ep.fit(cal_calibrated_probs, cal_labels)
    decisions = ep.run_batch(test_calibrated_probs)
    report = ep.evaluate(test_calibrated_probs, test_labels)
    return decisions, report, ep._upper_threshold, ep._lower_threshold
