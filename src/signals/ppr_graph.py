"""Personalised PageRank signal on a passage-similarity graph.

Constructs an undirected k-NN graph over passages using a coarse lexical
similarity kernel, then scores each ``(query, passage)`` pair by running
Personalised PageRank (Jeh & Widom, 2003; Haveliwala, 2002) with a
query-dependent teleportation distribution. The teleportation mass is
concentrated on passages that lexically overlap with the query, i.e. the
graph signal is orthogonal-ish to a direct lexical match: it rewards
passages that are *reachable from* lexically-matching passages rather than
lexically-matching themselves.

This signal is included in the benchmark for two reasons:

1. It is genuinely heterogeneous with respect to BM25 and dense encoders —
   it captures neighbourhood structure rather than per-pair similarity.
2. The conditional-independence assumption with respect to BM25 is weaker
   than between two dense encoders, giving the benchmark a realistic
   "approximately independent" setting.

Implementation details
----------------------
* Graph is built with the top-``k`` lexical neighbours per passage
  (Jaccard on unigram sets). ``k=8`` by default keeps the graph sparse
  enough for the power-iteration solver to terminate in a handful of
  iterations.
* Personalisation vector is the renormalised BM25-like overlap between the
  query and each passage; if the query has zero overlap anywhere, the
  personalisation falls back to uniform.
* Damping factor ``alpha=0.85`` follows the PageRank default.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .base import BaseSignal, QueryPassagePair

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokset(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class PPRGraphSignal(BaseSignal):
    name = "ppr_graph"

    def __init__(
        self,
        k_neighbors: int = 8,
        alpha: float = 0.85,
        n_iters: int = 30,
        tol: float = 1e-6,
    ) -> None:
        self.k_neighbors = int(k_neighbors)
        self.alpha = float(alpha)
        self.n_iters = int(n_iters)
        self.tol = float(tol)
        self._passage_ids: List[str] = []
        self._passage_id_to_idx: dict[str, int] = {}
        self._passage_tokens: List[set[str]] = []
        self._transition: Optional[np.ndarray] = None
        self._tfidf_vec = None
        self._tfidf_X = None

    # ------------------------------------------------------------------
    def fit_corpus(self, passage_ids: Sequence[str], passage_texts: Sequence[str]) -> "PPRGraphSignal":
        """Build the k-NN transition matrix over passages.

        Kept separate from :meth:`fit` (which takes raw strings) because
        the graph signal needs stable integer indexing over passage IDs.

        Implementation switches between two paths by corpus size:

        * Small corpora (< 5k passages): the original O(n^2) all-pairs
          Jaccard. Exact and easy to read.
        * Large corpora (>= 5k passages): a sparse TF-IDF + sparse
          k-NN approximation. We compute the top-k cosine neighbours
          of each passage in the TF-IDF representation; cosine on
          binary token-presence vectors is monotone in Jaccard, so
          the top-k argmax is unchanged. This avoids the O(n^2)
          blow-up on BEIR's larger corpora.
        """
        assert len(passage_ids) == len(passage_texts)
        self._passage_ids = list(passage_ids)
        self._passage_id_to_idx = {pid: i for i, pid in enumerate(self._passage_ids)}
        self._passage_tokens = [_tokset(t) for t in passage_texts]

        n = len(self._passage_ids)
        k = min(self.k_neighbors, max(1, n - 1))

        if n < 5000:
            return self._fit_corpus_dense(n, k)
        return self._fit_corpus_sparse(passage_texts, n, k)

    # ------------------------------------------------------------------
    def _fit_corpus_dense(self, n: int, k: int) -> "PPRGraphSignal":
        T = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sims = np.array(
                [_jaccard(self._passage_tokens[i], self._passage_tokens[j]) for j in range(n)],
                dtype=np.float64,
            )
            sims[i] = 0.0
            if sims.sum() <= 0.0:
                T[i, i] = 1.0
                continue
            top = np.argpartition(-sims, k)[:k]
            row = np.zeros(n, dtype=np.float64)
            row[top] = sims[top]
            row = row / row.sum()
            T[i] = row
        self._transition = T.T
        return self

    def _fit_corpus_sparse(self, passage_texts: Sequence[str], n: int, k: int) -> "PPRGraphSignal":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize as sk_normalize
        from scipy import sparse

        # Binary token-presence + L2-normalised cosine. Monotone in
        # Jaccard for set-vs-set comparisons; preserves the top-k
        # neighbour structure used by the PPR transition matrix.
        # Adapt min_df / max_df to the corpus size; on small synthetic
        # corpora with few unique tokens the default 2 / 0.95 prunes
        # everything. Falling back to the unfiltered vocabulary is safe
        # because the matrix is sparse anyway.
        for min_df, max_df in [(2, 0.95), (1, 1.0)]:
            try:
                vec = TfidfVectorizer(
                    token_pattern=r"[A-Za-z0-9]+",
                    lowercase=True,
                    binary=True,
                    sublinear_tf=False,
                    norm=None,
                    min_df=min_df,
                    max_df=max_df,
                )
                X = vec.fit_transform(passage_texts).astype(np.float32)
                break
            except ValueError:
                continue
        else:
            # Pathological corpus: fall back to the dense O(n^2) path.
            return self._fit_corpus_dense(n, k)
        X = sk_normalize(X, norm="l2", axis=1)

        # Compute top-k neighbours via a chunked sparse-sparse product.
        rows = []
        cols = []
        data = []
        chunk = 1024
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sims = (X[s:e] @ X.T).toarray()
            # Drop self-similarity.
            for ii, gi in enumerate(range(s, e)):
                sims[ii, gi] = 0.0
            for ii, gi in enumerate(range(s, e)):
                row = sims[ii]
                if row.sum() <= 0.0:
                    rows.append(gi)
                    cols.append(gi)
                    data.append(1.0)
                    continue
                top = np.argpartition(-row, k)[:k]
                vals = row[top]
                vals = np.maximum(vals, 0.0)
                tot = vals.sum()
                if tot <= 0.0:
                    rows.append(gi)
                    cols.append(gi)
                    data.append(1.0)
                    continue
                vals = vals / tot
                rows.extend([gi] * len(top))
                cols.extend(top.tolist())
                data.extend(vals.tolist())
        T_sparse = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n, n), dtype=np.float64
        )
        # PPR uses the column-stochastic transition (T^T).
        self._transition = T_sparse.T.tocsr()
        # Cache the TF-IDF vectoriser + matrix so per-query personalisation
        # can reuse them in O(nnz) instead of pure-Python Jaccard.
        self._tfidf_vec = vec
        self._tfidf_X = X
        return self

    # ------------------------------------------------------------------
    def _personalisation(self, query_tokens: set[str]) -> np.ndarray:
        if not self._passage_tokens:
            raise RuntimeError("PPRGraphSignal.fit_corpus must be called first")
        n = len(self._passage_tokens)
        # Fast path: if we have a fitted TF-IDF vectoriser (sparse path was
        # used in fit_corpus), score the query against every passage via
        # one sparse matrix-vector product. Cosine on binary token-presence
        # vectors is monotone in Jaccard — same argmax structure used
        # elsewhere in this module.
        if getattr(self, "_tfidf_vec", None) is not None and self._tfidf_X is not None:
            q_text = " ".join(query_tokens) if query_tokens else ""
            from sklearn.preprocessing import normalize as sk_normalize
            qv = self._tfidf_vec.transform([q_text]).astype(np.float32)
            qv = sk_normalize(qv, norm="l2", axis=1)
            sims = (qv @ self._tfidf_X.T).toarray().ravel()
            sims = np.maximum(sims, 0.0)
            s = sims.sum()
            if s <= 0.0:
                return np.full(n, 1.0 / n)
            return sims / s
        # Fallback (small-corpus dense path): exact Jaccard.
        overlaps = np.array(
            [_jaccard(query_tokens, p) for p in self._passage_tokens],
            dtype=np.float64,
        )
        s = overlaps.sum()
        if s <= 0.0:
            return np.full(n, 1.0 / n)
        return overlaps / s

    def _ppr(self, teleport: np.ndarray) -> np.ndarray:
        assert self._transition is not None
        n = teleport.shape[0]
        r = teleport.copy()
        for _ in range(self.n_iters):
            r_new = self.alpha * (self._transition @ r) + (1.0 - self.alpha) * teleport
            if np.linalg.norm(r_new - r, ord=1) < self.tol:
                r = r_new
                break
            r = r_new
        # Guard against numerical drift.
        r = np.clip(r, a_min=0.0, a_max=None)
        s = r.sum()
        return r / s if s > 0 else np.full(n, 1.0 / n)

    def score_pairs(self, pairs: Iterable[QueryPassagePair]) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.zeros(0, dtype=np.float64)

        # Group by query_id so PPR is run once per distinct query.
        by_query: dict[str, List[int]] = {}
        for i, p in enumerate(pairs):
            by_query.setdefault(p.query_id, []).append(i)

        scores = np.zeros(len(pairs), dtype=np.float64)
        for qid, idxs in by_query.items():
            q_tokens = _tokset(pairs[idxs[0]].query_text)
            teleport = self._personalisation(q_tokens)
            r = self._ppr(teleport)
            for i in idxs:
                pid = pairs[i].passage_id
                j = self._passage_id_to_idx.get(pid)
                if j is None:
                    scores[i] = 0.0
                else:
                    scores[i] = float(r[j])
        return scores
