"""Signal-dependence diagnostic.

We measure the conditional correlation of per-signal calibrated logits
given the relevance label ``Y``. Under the conditional-independence
assumption invoked by Theorem 1, ``corr(logit(p_i), logit(p_j) | Y=k) = 0``
for all ``i != j`` and ``k``. Any significant deviation is evidence that
the parametric CalFuse form is biased, and the learned variant should be
preferred.

Test statistic
--------------
For each pair of signals and each label class, compute the Pearson
correlation of calibrated logits over the subset of pairs with that
label. Fisher's z-transform gives an approximate ``N(0, 1 / (n - 3))``
null distribution. We combine tests across ``(i, j, k)`` triples via a
Holm-Bonferroni correction so the family-wise error rate matches the
declared ``alpha``. The scalar statistic returned is the maximum
absolute Fisher-z value across the family, and the flag is True iff
any of the Holm-corrected p-values falls below ``alpha``.

Returning both the statistic and the binary flag lets CalFuse use the
statistic as a soft signal for method selection while experimenters
retain the option to use the flag as a hard test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    sx = x.std()
    sy = y.std()
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(c) else c


def _fisher_z(r: float) -> float:
    r = max(-0.999999, min(0.999999, r))
    return 0.5 * np.log((1 + r) / (1 - r))


def _holm_bonferroni(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Return a boolean mask of rejections under Holm-Bonferroni."""
    n = pvals.size
    order = np.argsort(pvals)
    reject = np.zeros(n, dtype=bool)
    for rank, i in enumerate(order):
        thresh = alpha / (n - rank)
        if pvals[i] <= thresh:
            reject[i] = True
        else:
            break
    return reject


@dataclass
class SignalDependenceResult:
    statistic: float
    p_value: float
    reject: bool
    per_pair_correlations: np.ndarray  # shape (n_signals, n_signals)
    # Off-diagonal-mass score (sign-aware): a strictly more useful
    # statistic than max marginal correlation for predicting when the
    # parametric CalFuse rule is biased. Proposition 2 shows that the
    # first-order log-odds bias of the parametric rule is half the
    # off-diagonal sum of ``Sigma^{(1)} - Sigma^{(0)}``; this field
    # estimates that quantity directly. Empirical correlation of this
    # score with the Parametric-vs-Copula ECE gap is reported in
    # Section~\ref{sec:results}.
    off_diagonal_mass: float = 0.0


def signal_dependence_test(
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
) -> SignalDependenceResult:
    """Test conditional independence of signals given ``Y``.

    Parameters
    ----------
    calibrated_probs
        ``(n_pairs, n_signals)`` matrix of per-signal calibrated
        probabilities.
    labels
        Binary relevance labels.
    alpha
        Family-wise error rate for the Holm-Bonferroni correction.
    """
    from scipy.stats import norm

    L = _logit(np.asarray(calibrated_probs, dtype=np.float64))
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    n, d = L.shape
    C = np.zeros((d, d), dtype=np.float64)

    stats_list = []
    pvals_list = []
    pairs_list = []

    for k in (0, 1):
        mask = y == k
        if mask.sum() < 3:
            continue
        sub = L[mask]
        n_k = sub.shape[0]
        for i in range(d):
            for j in range(i + 1, d):
                r = _pearson_corr(sub[:, i], sub[:, j])
                C[i, j] = max(C[i, j], abs(r))
                C[j, i] = C[i, j]
                z = _fisher_z(r) * np.sqrt(max(1, n_k - 3))
                p = 2.0 * (1.0 - norm.cdf(abs(z)))
                stats_list.append(abs(z))
                pvals_list.append(p)
                pairs_list.append((i, j, k))

    if not stats_list:
        return SignalDependenceResult(0.0, 1.0, False, C, 0.0)

    stats = np.array(stats_list)
    pvals = np.array(pvals_list)
    reject_mask = _holm_bonferroni(pvals, alpha)

    # Off-diagonal mass of Sigma^{(1)} - Sigma^{(0)} on calibrated
    # logits (Prop 2). Sign-preserving sum of off-diagonal entries.
    off_mass = 0.0
    if (y == 0).sum() >= 2 and (y == 1).sum() >= 2:
        S0 = np.cov(L[y == 0], rowvar=False)
        S1 = np.cov(L[y == 1], rowvar=False)
        if S0.ndim >= 2:
            D = S1 - S0
            np.fill_diagonal(D, 0.0)
            # Half the off-diagonal sum: the bias term in Prop 2.
            off_mass = 0.5 * float(D.sum())

    return SignalDependenceResult(
        statistic=float(stats.max()),
        p_value=float(pvals.min()),
        reject=bool(reject_mask.any()),
        per_pair_correlations=C,
        off_diagonal_mass=off_mass,
    )
