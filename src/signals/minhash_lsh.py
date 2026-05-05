"""MinHash / Jaccard similarity signal.

Implements a compact MinHash sketch (Broder, 1997) and returns the estimated
Jaccard similarity between query and passage token-shingle sets. MinHash is
included in the benchmark as a *structural* lexical signal: it is
near-perfectly correlated with Jaccard overlap but has calibrated variance
depending on the number of permutations, so the calibration-fit step has a
non-trivial job mapping raw Jaccard into a probability of relevance.

We do not take the dependency on ``datasketch`` here — a NumPy-only
implementation keeps the core benchmark reproducible without external
packages. ``datasketch`` can be used as a drop-in at larger scales.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from .base import BaseSignal, QueryPassagePair

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _shingles(text: str, n: int = 2) -> list[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    if len(toks) < n:
        return toks
    return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]


class MinHashLSHSignal(BaseSignal):
    name = "minhash_lsh"

    def __init__(self, n_perm: int = 64, shingle_n: int = 2, seed: int = 13) -> None:
        self.n_perm = int(n_perm)
        self.shingle_n = int(shingle_n)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        # Coefficients for 2-universal hashes over a large prime.
        self._prime = (1 << 61) - 1
        self._a = rng.integers(1, self._prime, size=self.n_perm, dtype=np.int64)
        self._b = rng.integers(0, self._prime, size=self.n_perm, dtype=np.int64)

    def _minhash(self, tokens: list[str]) -> np.ndarray:
        if not tokens:
            return np.full(self.n_perm, np.iinfo(np.int64).max, dtype=np.int64)
        # Python hash() is stable within a process; seed via PYTHONHASHSEED
        # is handled at the benchmark-construction layer.
        base = np.array([hash(t) & ((1 << 61) - 1) for t in tokens], dtype=np.int64)
        # shape: (n_perm, n_tokens)
        # ((a * h + b) mod p) broadcast; reduce min along axis=1.
        mixed = (self._a[:, None] * base[None, :] + self._b[:, None]) % self._prime
        return mixed.min(axis=1)

    def score_pairs(self, pairs: Iterable[QueryPassagePair]) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.zeros(0, dtype=np.float64)
        scores = np.zeros(len(pairs), dtype=np.float64)
        for i, p in enumerate(pairs):
            q_sh = _shingles(p.query_text, self.shingle_n)
            d_sh = _shingles(p.passage_text, self.shingle_n)
            qh = self._minhash(q_sh)
            dh = self._minhash(d_sh)
            scores[i] = float((qh == dh).mean())
        return scores
