"""BM25 retrieval signal.

A pure-NumPy implementation of Okapi BM25 (Robertson & Zaragoza, 2009) that
scores individual ``(query, passage)`` pairs rather than full rankings.
Corpus-level IDF is fit from the calibration partition of the benchmark only;
the test partition never contributes to IDF statistics, enforcing the
calibration-fit / test separation required by the benchmark protocol.

Why implement BM25 ourselves rather than calling ``rank_bm25`` or Pyserini?
We score *pairs*, not rankings. ``rank_bm25`` exposes a per-query score
interface but incurs quadratic overhead when many queries are scored against
an identical set of passages (the common case in a fused benchmark). A direct
NumPy implementation lets us precompute per-passage term vectors once.

Notes
-----
* Tokenization uses a simple lowercase word regex. The benchmark fixes this
  choice so that results are reproducible without a tokenizer dependency.
* BM25 scores are unbounded and distribution-dependent, so the raw score is
  *not* a probability. Calibration of BM25 output is handled downstream.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, List, Sequence

import numpy as np

from .base import BaseSignal, QueryPassagePair

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


class BM25Signal(BaseSignal):
    """Okapi BM25 scorer for ``(query, passage)`` pairs.

    Parameters
    ----------
    k1, b : standard BM25 hyperparameters. Defaults follow Robertson &
        Zaragoza (2009), ``k1=1.5``, ``b=0.75``. These are fixed for the
        benchmark; grid search over BM25 hyperparameters is intentionally out
        of scope — calibration is applied downstream and absorbs mild
        miscalibration introduced by suboptimal ``k1``/``b``.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, corpus: Sequence[str]) -> "BM25Signal":
        tokenised = [_tokenize(doc) for doc in corpus]
        n_docs = max(1, len(tokenised))
        doc_lengths = np.array([len(d) for d in tokenised], dtype=np.float64)
        self._avgdl = float(doc_lengths.mean()) if doc_lengths.size else 1.0

        df: dict[str, int] = defaultdict(int)
        for doc in tokenised:
            for term in set(doc):
                df[term] += 1

        # Standard BM25 IDF with the "plus-one" smoothing to avoid negative
        # weights for very common terms.
        self._idf = {
            term: math.log((n_docs - dfi + 0.5) / (dfi + 0.5) + 1.0)
            for term, dfi in df.items()
        }
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------
    def score_pairs(self, pairs: Iterable[QueryPassagePair]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BM25Signal.fit must be called before score_pairs")

        pairs = list(pairs)
        if not pairs:
            return np.zeros(0, dtype=np.float64)

        scores = np.zeros(len(pairs), dtype=np.float64)
        for i, p in enumerate(pairs):
            q_tokens = _tokenize(p.query_text)
            d_tokens = _tokenize(p.passage_text)
            if not d_tokens:
                continue
            tf = Counter(d_tokens)
            dl = len(d_tokens)
            norm = 1.0 - self.b + self.b * (dl / max(1e-9, self._avgdl))
            s = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                t = tf[term]
                s += idf * (t * (self.k1 + 1.0)) / (t + self.k1 * norm)
            scores[i] = s
        return scores
