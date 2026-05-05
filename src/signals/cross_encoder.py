"""Cross-encoder rerank signal.

Wraps a cross-encoder (Nogueira & Cho, 2019, "Passage Re-ranking with BERT";
MiniLM variants from Reimers & Gurevych, 2020) through
``sentence-transformers.CrossEncoder``. Returns raw logits — *not* sigmoid
probabilities — so that downstream calibration can learn the correct
sigmoid temperature rather than inheriting a preset one.

Why raw logits, not sigmoid? Guo et al. (2017), "On Calibration of Modern
Neural Networks", show that vanilla softmax/sigmoid outputs of trained
classifiers are systematically miscalibrated — typically overconfident for
in-distribution examples and arbitrarily scaled for out-of-distribution
examples. Feeding raw logits into Platt / isotonic / temperature scaling
yields a better-calibrated probability than trusting the model's own sigmoid.

Fallback behaviour mirrors the dense signals: when the real package is
missing, a deterministic blend of BM25-like token overlap and hashed cosine
is used so that smoke tests and CI runs remain reproducible.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import numpy as np

from .base import BaseSignal, QueryPassagePair


def _offline_ce_logit(query: str, passage: str, seed: int = 7) -> float:
    """Deterministic pseudo-logit for the fallback path.

    Combines lexical overlap with a seeded noise term. Stays on a roughly
    ``[-3, 3]`` scale so downstream logistic calibration has something
    meaningful to fit.
    """
    q_tokens = set(query.lower().split())
    p_tokens = set(passage.lower().split())
    if not q_tokens:
        overlap = 0.0
    else:
        overlap = len(q_tokens & p_tokens) / len(q_tokens)
    h = hashlib.sha1(f"{seed}:{query}|{passage}".encode()).digest()
    noise = (int.from_bytes(h[:4], "big") / 2**32 - 0.5) * 0.6
    return 4.0 * (overlap - 0.25) + noise


class CrossEncoderSignal(BaseSignal):
    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        force_fallback: bool = False,
        seed: int = 7,
    ) -> None:
        self.model_name = model_name
        self.force_fallback = force_fallback
        self.seed = seed
        self._model: Optional[object] = None
        self._using_fallback: bool = force_fallback
        self._try_load_real_model()

    def _try_load_real_model(self) -> None:
        if self.force_fallback:
            return
        try:  # pragma: no cover
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.model_name)
            self._using_fallback = False
        except Exception:  # noqa: BLE001
            self._model = None
            self._using_fallback = True

    def score_pairs(self, pairs: Iterable[QueryPassagePair]) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.zeros(0, dtype=np.float64)

        if self._using_fallback or self._model is None:
            return np.array(
                [_offline_ce_logit(p.query_text, p.passage_text, self.seed) for p in pairs],
                dtype=np.float64,
            )

        inputs = [(p.query_text, p.passage_text) for p in pairs]
        # activation_fct=None -> return raw logits (Guo et al., 2017).
        logits = self._model.predict(inputs, activation_fct=None)  # type: ignore[attr-defined]
        return np.asarray(logits, dtype=np.float64).reshape(-1)
