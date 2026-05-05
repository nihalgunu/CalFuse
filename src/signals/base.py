"""Base class for retrieval signals.

A signal is any function that assigns a real-valued score to a ``(query,
passage)`` pair. The only contract is the presence of ``score_pairs``. Signals
may optionally implement ``fit`` when they require corpus statistics (e.g. IDF
for BM25, PPR matrices for graph signals). Fitting must be idempotent and must
not consume information from the held-out test partition — enforcement is at
the benchmark-construction layer (see :mod:`eval.build_benchmark`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class QueryPassagePair:
    query_id: str
    passage_id: str
    query_text: str
    passage_text: str


class BaseSignal(ABC):
    """Abstract retrieval signal.

    Subclasses must implement :meth:`score_pairs`. Subclasses *may* override
    :meth:`fit` if corpus statistics are required prior to scoring.
    """

    name: str = "base"

    def fit(self, corpus: Sequence[str]) -> "BaseSignal":  # noqa: D401 - small default
        """Optional corpus-level fit; default is a no-op."""
        return self

    @abstractmethod
    def score_pairs(self, pairs: Iterable[QueryPassagePair]) -> np.ndarray:
        """Return a 1-D float array of raw scores aligned with ``pairs``."""
        raise NotImplementedError
