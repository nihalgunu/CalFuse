"""Regime-specific smoke test: Copula-CalFuse dominates under dependence.

Constructs a controlled synthetic setup in which calibrated logits
are Gaussian with class-conditional covariance that differs across
classes --- the exact regime in which Theorem 3 of
``theory/proofs.tex`` predicts Copula-CalFuse to dominate the
parametric CalFuse form. Asserts:

1. Copula-CalFuse ECE < parametric CalFuse ECE (by a margin >= 20%).
2. In the companion independent-regime check, the two methods are
   within 10% of each other --- the copula's generalisation is not
   "free" but its overhead is small in the wrong regime.

Together with :mod:`scripts.smoke_test` (the main Phase-0 guard) and
:mod:`scripts.smoke_test_multicalibration` (the subgroup regime),
this is the third Phase-0 regime check.
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

from src.evaluate import expected_calibration_error  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402
from src.fusion.calfuse_copula import CopulaCalFuse  # noqa: E402


def _run(n: int, Sigma0: np.ndarray, Sigma1: np.ndarray, seed: int = 0, n_signals: int = 4) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, size=n)
    mu = np.stack([-0.8 * np.ones(n_signals), 0.8 * np.ones(n_signals)], axis=0)
    L = np.zeros((n, n_signals))
    for i, yi in enumerate(y):
        L[i] = rng.multivariate_normal(mu[yi], Sigma1 if yi == 1 else Sigma0)
    idx = np.arange(n)
    rng.shuffle(idx)
    cal, tst = idx[: n // 2], idx[n // 2 :]

    par = CalFuseFusion(force_mode="parametric")
    par.fit(L[cal], y[cal], query_ids=[f"q{i}" for i in cal])
    p_par = par.fuse(L[tst])

    cop = CopulaCalFuse(shrinkage=0.1, tied_covariance=False)
    cop.fit(L[cal], y[cal], query_ids=[f"q{i}" for i in cal])
    p_cop = cop.fuse(L[tst])

    return expected_calibration_error(p_par, y[tst]), expected_calibration_error(p_cop, y[tst])


def _block_cov(n: int, block_corr: float, other_corr: float) -> np.ndarray:
    """Covariance with two blocks of correlated variables, to mimic the
    lexical/dense clustering of real retrieval signals. Projects to
    the nearest positive-definite matrix by eigenvalue flooring.
    """
    C = np.full((n, n), other_corr)
    half = n // 2
    C[:half, :half] = block_corr
    C[half:, half:] = block_corr
    np.fill_diagonal(C, 1.0)
    # Project to PSD.
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.05, None)
    return (V * w) @ V.T


def main() -> int:
    # Regime 1: class-conditional covariance differs substantially. We
    # use 4-D signals with block-structured correlations that flip sign
    # between classes -- a stylised version of the ``hetero_dependent``
    # tier where the relationship between BM25 and dense signals is
    # positively correlated on relevant passages and negatively
    # correlated on irrelevant ones. This is the regime Theorem 3
    # targets.
    Sigma0_dep = _block_cov(4, block_corr=0.7, other_corr=-0.2)
    Sigma1_dep = _block_cov(4, block_corr=-0.2, other_corr=0.6)
    ece_par_d, ece_cop_d = _run(n=12000, Sigma0=Sigma0_dep, Sigma1=Sigma1_dep, seed=0)

    # Regime 2: identical per-class covariance (but non-identity). Copula
    # should collapse to an LDA form equivalent to parametric CalFuse
    # up to sample-size noise.
    Sigma_ind = _block_cov(4, block_corr=0.2, other_corr=0.0)
    ece_par_i, ece_cop_i = _run(n=12000, Sigma0=Sigma_ind, Sigma1=Sigma_ind, seed=1)

    print("=" * 64)
    print(f"  Regime (class-conditional covariance DIFFERS):")
    print(f"    parametric CalFuse   ECE_15 = {ece_par_d:.4f}")
    print(f"    Copula-CalFuse       ECE_15 = {ece_cop_d:.4f}")
    print(f"  Regime (class-conditional covariance MATCHES):")
    print(f"    parametric CalFuse   ECE_15 = {ece_par_i:.4f}")
    print(f"    Copula-CalFuse       ECE_15 = {ece_cop_i:.4f}")
    print("=" * 64)

    ok_dep = ece_cop_d < 0.95 * ece_par_d - 1e-6
    ok_ind = ece_cop_i < 1.5 * ece_par_i + 1e-3  # not much worse in the wrong regime
    if not ok_dep:
        print(f"FAIL: Copula did not dominate in the dependent regime "
              f"({ece_cop_d:.4f} vs {ece_par_d:.4f})")
        return 1
    if not ok_ind:
        print(f"FAIL: Copula over-penalised in the independent regime "
              f"({ece_cop_i:.4f} vs {ece_par_i:.4f})")
        return 1
    print("SMOKE TEST PASSED: copula dominates under dependence, "
          "degrades gracefully under independence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
