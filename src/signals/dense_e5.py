"""E5 dense retrieval signal.

Wraps the E5 encoder family (Wang et al., 2022, "Text Embeddings by Weakly-
Supervised Contrastive Pre-training") via ``sentence-transformers`` when
available. Uses the same deterministic hashed-embedding fallback as
:class:`DenseBGESignal`, but seeded differently so the two offline surrogates
are not identical.

Relative to BGE, E5 is trained with a different contrastive objective and on
a partially overlapping corpus. In the *heterogeneous-dependent* tier of the
benchmark we pair E5 with BGE specifically to induce the conditional
dependence failure mode that CalFuse must handle.

E5 requires input prefixes (``query:`` / ``passage:``) at inference time;
we prepend them here so calling code does not have to.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from .base import BaseSignal, QueryPassagePair
from .dense_bge import _hashed_embedding


class DenseE5Signal(BaseSignal):
    name = "dense_e5"

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        embedding_dim_fallback: int = 64,
        seed: int = 1,
        force_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim_fallback = embedding_dim_fallback
        self.seed = seed
        self.force_fallback = force_fallback
        self._model: Optional[object] = None
        self._using_fallback: bool = force_fallback
        self._try_load_real_model()

    def _try_load_real_model(self) -> None:
        if self.force_fallback:
            return
        try:  # pragma: no cover
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
            self._using_fallback = False
        except Exception:  # noqa: BLE001
            self._model = None
            self._using_fallback = True

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        prefixed = [f"{prefix}: {t}" for t in texts]
        if self._using_fallback or self._model is None:
            return np.stack(
                [
                    _hashed_embedding(t, self.embedding_dim_fallback, self.seed)
                    for t in prefixed
                ]
            )
        emb = self._model.encode(prefixed, normalize_embeddings=True)  # type: ignore[attr-defined]
        return np.asarray(emb, dtype=np.float64)

    def score_pairs(self, pairs: Iterable[QueryPassagePair]) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.zeros(0, dtype=np.float64)

        # Dedup before encoding — candidate pools reuse queries and passages heavily.
        uniq_q: dict[str, int] = {}
        uniq_p: dict[str, int] = {}
        q_texts: list[str] = []
        p_texts: list[str] = []
        for pair in pairs:
            if pair.query_text not in uniq_q:
                uniq_q[pair.query_text] = len(q_texts)
                q_texts.append(pair.query_text)
            if pair.passage_text not in uniq_p:
                uniq_p[pair.passage_text] = len(p_texts)
                p_texts.append(pair.passage_text)

        q_emb_uniq = self._encode(q_texts, prefix="query")
        p_emb_uniq = self._encode(p_texts, prefix="passage")

        q_emb = q_emb_uniq[[uniq_q[pair.query_text] for pair in pairs]]
        d_emb = p_emb_uniq[[uniq_p[pair.passage_text] for pair in pairs]]

        denom = np.linalg.norm(q_emb, axis=1) * np.linalg.norm(d_emb, axis=1)
        denom = np.maximum(denom, 1e-12)
        return np.sum(q_emb * d_emb, axis=1) / denom
