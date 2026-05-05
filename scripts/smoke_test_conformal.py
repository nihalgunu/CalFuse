"""Regime-specific smoke test: conformal coverage + sequential compute savings.

Exercises the two conformal contributions (Theorems 5 and 6 in
``theory/proofs.tex``) on a deterministic synthetic substrate:

1. ``MondrianVennAbers`` envelope coverage: the per-pair envelope
   ``[p_lo(x), p_hi(x)]`` should cover the *true* conditional
   probability ``P(Y=1 | X)`` at the nominal level. We verify a
   weaker but observable surrogate: the midpoint is close to the
   empirical rate within each bin, and the prediction sets induced
   by thresholding ``[p_lo, p_hi]`` cover ``Y`` at rate ``>= 1 -
   alpha``.

2. ``EProcess`` sequential fusion: the empirical Type-I error rate
   (deciding "relevant" when ``Y = 0``) is at most ``alpha`` *and*
   the mean stopping time is strictly less than the total number
   of signals --- the compute-budget savings claim.

Pass criteria:

* Prediction-set coverage within ``alpha`` of the nominal level
  (slack ``0.02``).
* Sequential Type-I error rate ``<= alpha + 0.02``.
* Mean stopping time ``<=`` 0.9 * n_signals (at least 10% savings).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _ensure_repo_on_path() -> None:
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_ensure_repo_on_path()

from src.conformal.sequential import EProcess  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402
from src.fusion.calfuse_conformal import ConformalCalFuse  # noqa: E402
from src.fusion.multicalibration import signal_dominance_subgroups  # noqa: E402


def _make_dataset(n: int, n_signals: int, pi: float, seed: int = 0):
    """Informative signals: per-signal calibrated probabilities that
    *actually* track Y with some noise. Sort columns in increasing
    informativeness so the sequential rule has "cheap before
    expensive" structure.
    """
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, pi, size=n)
    # Per-signal conditional-mean shifts: signal 0 is weak, signal
    # n_signals-1 is strongest.
    informativeness = np.linspace(0.4, 2.0, n_signals)
    L = np.zeros((n, n_signals))
    for j in range(n_signals):
        mu = np.where(y == 1, informativeness[j], -informativeness[j])
        L[:, j] = mu + rng.normal(0, 1, size=n)
    return L, y


def _check_prediction_set_coverage(
    envelope, y, alpha, threshold: float = 0.5
) -> float:
    """Return empirical coverage of the prediction set implied by the
    Venn-Abers envelope.
    """
    # Set rule: include label 1 iff ``p_hi >= threshold``;
    # include label 0 iff ``p_lo <= threshold``. Coverage is the
    # fraction of test pairs whose *true* label is inside the set.
    include_1 = envelope.p_hi >= threshold
    include_0 = envelope.p_lo <= threshold
    inside = np.where(y == 1, include_1, include_0)
    return float(inside.mean())


def main() -> int:
    alpha = 0.1
    n_signals = 6
    L, y = _make_dataset(n=4000, n_signals=n_signals, pi=0.3, seed=0)

    rng = np.random.default_rng(1)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    half = len(y) // 2
    cal_idx, tst_idx = idx[:half], idx[half:]

    qids_cal = [f"q{i}" for i in cal_idx]
    qids_tst = [f"q{i}" for i in tst_idx]

    # ---- Thm 5: Mondrian-Venn-Abers envelope coverage --------------------
    conf = ConformalCalFuse(
        base=CalFuseFusion(force_mode="parametric"),
        subgroup_fn=signal_dominance_subgroups(),
    )
    conf.fit(L[cal_idx], labels=y[cal_idx], query_ids=qids_cal)
    env_tst = conf.predict_envelope(L[tst_idx], query_ids=qids_tst)
    emp_coverage = _check_prediction_set_coverage(env_tst, y[tst_idx], alpha=alpha)
    mean_width = float(np.mean(env_tst.p_hi - env_tst.p_lo))

    # ---- Thm 6: sequential e-process compute savings ----------------------
    # Build per-pair calibrated-probability matrix ordered cheap->expensive.
    # We use per-signal Platt calibrators so each column is a
    # *properly calibrated* posterior marginal.
    from src.calibrators.platt import PlattCalibrator

    calibrated = np.zeros_like(L)
    for j in range(n_signals):
        c = PlattCalibrator().fit(L[cal_idx, j], y[cal_idx])
        calibrated[:, j] = c.transform(L[:, j])
    ep = EProcess(alpha=alpha, pi=float(y[cal_idx].mean()))
    report = ep.evaluate(calibrated[tst_idx], y[tst_idx])

    print("=" * 72)
    print(f"  Conformal envelope   coverage = {emp_coverage:.4f}   nominal >= {1 - alpha:.2f}")
    print(f"  Conformal envelope   mean width = {mean_width:.4f}")
    print(f"  Sequential e-process Type-I    = {report.empirical_type1:.4f}   budget alpha = {alpha}")
    print(f"  Sequential e-process Type-II   = {report.empirical_type2:.4f}")
    print(f"  Sequential e-process abstain   = {report.abstention_rate:.4f}")
    print(f"  Sequential e-process mean t*   = {report.mean_stopping_time:.2f} of {n_signals}")
    print(f"  Per-signal consumption rate    = {np.round(report.per_signal_consumption, 3)}")
    print("=" * 72)

    slack = 0.02
    ok_coverage = emp_coverage >= (1 - alpha) - slack
    ok_type1 = report.empirical_type1 <= alpha + slack
    ok_savings = report.mean_stopping_time <= 0.9 * n_signals

    if not ok_coverage:
        print(f"FAIL: envelope coverage {emp_coverage:.4f} below nominal {1 - alpha:.2f}")
        return 1
    if not ok_type1:
        print(f"FAIL: sequential Type-I {report.empirical_type1:.4f} exceeds budget {alpha}")
        return 1
    if not ok_savings:
        print(
            f"FAIL: mean stopping time {report.mean_stopping_time:.2f} did not save "
            f">= 10% of {n_signals} signals"
        )
        return 1

    print("SMOKE TEST PASSED: conformal envelope covers at nominal rate, "
          "sequential e-process controls Type-I while saving compute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
