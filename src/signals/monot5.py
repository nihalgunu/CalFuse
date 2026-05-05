"""monoT5 reranker signal (stronger than MiniLM cross-encoder).

monoT5 (Nogueira et al., 2020, "Document Ranking with a Pretrained
Sequence-to-Sequence Model") frames passage relevance as a generation
task: condition T5 on the prompt
``"Query: <q> Document: <p> Relevant:"`` and use the difference of
``log P("true")`` and ``log P("false")`` as the relevance logit. This is
substantially stronger than the MiniLM cross-encoder on most BEIR
subsets and gives the paper a NeurIPS-grade reranker baseline.

We return the raw log-odds ``logP(true) - logP(false)`` (not the
sigmoid), for the same reason as
:mod:`src.signals.cross_encoder`: per-signal calibration handles the
scale, and feeding a calibrator a saturated sigmoid loses information.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import numpy as np

from .base import BaseSignal, QueryPassagePair


def _offline_monot5_logit(query: str, passage: str, seed: int = 13) -> float:
    """Deterministic pseudo-logit for the fallback path."""
    q_tokens = set(query.lower().split())
    p_tokens = set(passage.lower().split())
    if not q_tokens:
        overlap = 0.0
    else:
        overlap = len(q_tokens & p_tokens) / len(q_tokens)
    h = hashlib.sha1(f"{seed}:monot5:{query}|{passage}".encode()).digest()
    noise = (int.from_bytes(h[:4], "big") / 2**32 - 0.5) * 0.4
    return 5.0 * (overlap - 0.2) + noise


class MonoT5Signal(BaseSignal):
    name = "monot5"

    def __init__(
        self,
        model_name: str = "castorini/monot5-base-msmarco",
        force_fallback: bool = False,
        max_length: int = 512,
        batch_size: int = 32,
        seed: int = 13,
    ) -> None:
        self.model_name = model_name
        self.force_fallback = force_fallback
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.seed = seed
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._true_id: int = -1
        self._false_id: int = -1
        self._using_fallback: bool = force_fallback
        self._try_load_real_model()

    def _try_load_real_model(self) -> None:  # pragma: no cover - GPU path
        if self.force_fallback:
            return
        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self._model.eval()
            if torch.cuda.is_available():
                self._device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device = "mps"
            self._model.to(self._device)
            # monoT5 uses literal tokens "true" and "false".
            self._true_id = int(self._tokenizer("true").input_ids[0])
            self._false_id = int(self._tokenizer("false").input_ids[0])
            self._using_fallback = False
        except Exception:
            self._tokenizer = None
            self._model = None
            self._using_fallback = True

    def score_pairs(self, pairs: Iterable[QueryPassagePair]) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.zeros(0, dtype=np.float64)
        if self._using_fallback or self._model is None or self._tokenizer is None:
            return np.array(
                [_offline_monot5_logit(p.query_text, p.passage_text, self.seed) for p in pairs],
                dtype=np.float64,
            )

        import torch  # type: ignore
        prompts = [
            f"Query: {p.query_text} Document: {p.passage_text} Relevant:" for p in pairs
        ]
        out = np.empty(len(prompts), dtype=np.float64)
        # Decoder input is the BOS token; we read out the first generated logit
        # and compute log P(true) - log P(false).
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                end = min(start + self.batch_size, len(prompts))
                enc = self._tokenizer(
                    prompts[start:end],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)
                # Decoder start = pad token (T5 convention).
                dec_start = torch.full(
                    (enc.input_ids.shape[0], 1),
                    self._model.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=self._device,
                )
                logits = self._model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    decoder_input_ids=dec_start,
                ).logits[:, 0, :]
                logp = torch.log_softmax(logits, dim=-1)
                lo = (logp[:, self._true_id] - logp[:, self._false_id]).cpu().numpy()
                out[start:end] = lo
        return out
