"""BEIR subset loader for the primary CalFuse evaluation.

Our claim is about calibration of fused retrieval scores on
realistic heterogeneous signals. The correct way to evaluate that
claim is on the established public benchmark the retrieval
community uses: BEIR (Thakur et~al., 2021). This module loads a
named BEIR subset and produces exactly the inputs the CalFuse
evaluation pipeline consumes.

What we do *not* do is claim a new benchmark. The controlled
simulations in :mod:`eval.build_benchmark` are regime-targeted
sanity checks for specific theorems; primary numbers come from
BEIR.

Substrates
----------
Supported subsets, all of which ship graded relevance labels that
binarise cleanly for ECE evaluation:

* ``nfcorpus``  --- medical IR, very small (~300 test queries).
* ``scifact``  --- fact verification over scientific claims.
* ``fiqa``  --- opinion-based financial QA.
* ``trec-covid``  --- COVID-19 literature retrieval.
* ``arguana``  --- argument retrieval.
* ``cqadupstack-english``  --- community QA duplicate detection.

Defaults follow the BEIR paper's "zero-shot" evaluation setting,
with one exception: because CalFuse is a \\emph{calibration}
method, it requires labelled data. We therefore use BEIR subsets
that ship non-trivial train or validation splits; subsets that
have only a test split (e.g.\\ NFCorpus) are evaluated via a
query-level split of the test partition.

Loading path
------------
We prefer ``datasets`` (HuggingFace) with the ``BeIR/*`` repos,
falling back to the ``beir`` package when HF is unavailable.
Downloaded content is cached under ``data/beir_subsets/<name>/``.

Candidate pool construction
---------------------------
Retrieval calibration is a pair-level problem and BEIR qrels are
sparse (most passages are unlabelled and assumed irrelevant). For
each query we construct a *candidate pool* of up to ``top_k_bm25``
BM25 hits plus all labelled-positive passages:

* positives: every passage with a non-zero qrel for the query
  (always included so the positive rate is non-zero);
* hard negatives: top BM25 hits among passages that are \\emph{not}
  labelled positive for that query --- matches the MS MARCO hard-
  negative mining protocol used in dense-retrieval training.

This candidate-pool construction is \\emph{standard}
(e.g.\\ Karpukhin et~al., 2020). The deviation from the BEIR
full-corpus retrieval setting is necessary because evaluating all
corpus-wide pairs is computationally prohibitive and because
calibration of unlabelled pairs is ill-defined.

Splits
------
By default we run a query-level 50 / 20 / 30 split on the test
queries (calibration / validation / test). Training queries are
unused in the zero-shot protocol to keep the comparison faithful
to prior work; ablations that fit calibrators on training-split
queries are reported separately in Phase 3.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


SUPPORTED = {
    "nfcorpus": "BeIR/nfcorpus",
    "scifact": "BeIR/scifact",
    "fiqa": "BeIR/fiqa",
    "trec-covid": "BeIR/trec-covid",
    "arguana": "BeIR/arguana",
    "scidocs": "BeIR/scidocs",
    "touche-2020": "BeIR/webis-touche2020",
    "quora": "BeIR/quora",
    # Additional regime-test subsets for Phase 2 of the regime
    # characterisation experiment.
    "trec-news": "BeIR/trec-news",
    "robust04": "BeIR/robust04",
    "cqadupstack-android": "BeIR/cqadupstack-android",
    "cqadupstack-gaming": "BeIR/cqadupstack-gaming",
    "cqadupstack-tex": "BeIR/cqadupstack-tex",
    "cqadupstack-mathematica": "BeIR/cqadupstack-mathematica",
}


@dataclass
class BEIRSubset:
    name: str
    queries: dict[str, str]  # qid -> query text
    corpus: dict[str, str]  # pid -> passage text
    qrels: dict[str, dict[str, int]]  # qid -> {pid: graded_relevance}


# ---------------------------------------------------------------------------
# download + parse
# ---------------------------------------------------------------------------
def _cache_dir(name: str) -> Path:
    p = Path(__file__).resolve().parent.parent / "data" / "beir_subsets" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_via_hf(name: str) -> BEIRSubset:
    """Load a BEIR subset via HuggingFace ``datasets``.

    Requires the ``datasets`` package and a working network
    connection on first use. Subsequent loads read from the
    HuggingFace cache. Raises ``RuntimeError`` if HF is unavailable
    so callers can fall back to the ``beir`` package.
    """
    from datasets import load_dataset  # type: ignore

    repo = SUPPORTED[name]
    cache = str(_cache_dir(name))

    corpus_ds = load_dataset(repo, "corpus", cache_dir=cache)["corpus"]
    queries_ds = load_dataset(repo, "queries", cache_dir=cache)["queries"]
    qrels_ds = load_dataset(f"{repo}-qrels", cache_dir=cache)

    # BEIR qrels on HF live in train / validation / test splits. We
    # concatenate validation+test for evaluation (calibration split
    # is carved out later) and keep train separate.
    qrels: dict[str, dict[str, int]] = {}
    for split in ("test", "validation", "dev"):
        if split in qrels_ds:
            for ex in qrels_ds[split]:
                qid = str(ex["query-id"])
                pid = str(ex["corpus-id"])
                rel = int(ex.get("score", 1))
                qrels.setdefault(qid, {})[pid] = rel

    corpus: dict[str, str] = {}
    for ex in corpus_ds:
        pid = str(ex["_id"])
        # Concatenate title + text when both are present.
        t = " ".join([ex.get("title") or "", ex.get("text") or ""]).strip()
        corpus[pid] = t

    queries: dict[str, str] = {}
    for ex in queries_ds:
        qid = str(ex["_id"])
        queries[qid] = ex.get("text") or ""
    # Keep only queries that have at least one relevance judgement.
    queries = {qid: q for qid, q in queries.items() if qid in qrels}

    return BEIRSubset(name=name, queries=queries, corpus=corpus, qrels=qrels)


def _load_via_beir(name: str) -> BEIRSubset:
    """Fallback to the ``beir`` package."""
    from beir import util  # type: ignore
    from beir.datasets.data_loader import GenericDataLoader  # type: ignore

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    out_dir = _cache_dir(name)
    data_path = util.download_and_unzip(url, str(out_dir))
    # GenericDataLoader returns (corpus, queries, qrels) keyed by id.
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    corpus_flat = {pid: f"{doc.get('title', '')} {doc.get('text', '')}".strip() for pid, doc in corpus.items()}
    return BEIRSubset(name=name, queries=queries, corpus=corpus_flat, qrels=qrels)


def load_beir(name: str) -> BEIRSubset:
    """Load a supported BEIR subset, preferring HF datasets."""
    if name not in SUPPORTED:
        raise ValueError(
            f"Unsupported BEIR subset {name!r}; supported: {sorted(SUPPORTED)}"
        )
    try:
        return _load_via_hf(name)
    except Exception as hf_err:
        try:
            return _load_via_beir(name)
        except Exception as beir_err:
            raise RuntimeError(
                f"Failed to load BEIR subset {name!r} via both HuggingFace "
                f"({hf_err}) and the beir package ({beir_err}). Run on an "
                "internet-connected machine on first use to populate the cache."
            )


# ---------------------------------------------------------------------------
# candidate-pool construction
# ---------------------------------------------------------------------------
def build_candidate_pool(
    subset: BEIRSubset,
    top_k_bm25: int = 100,
    max_negatives_per_query: int = 40,
    seed: int = 2026,
) -> tuple[list, list[float], list[str], list[int]]:
    """Materialise per-query candidate pools of labelled positives +
    BM25-mined hard negatives.

    Returns
    -------
    pairs : list[QueryPassagePair]
    graded : list[float]       -- graded relevance per pair (0 if unlabelled).
    qids   : list[str]         -- aligned row-wise with ``pairs``.
    cand_k : list[int]         -- per-query candidate-pool size; sum is
                                  ``len(pairs)``.
    """
    from src.signals.base import QueryPassagePair
    from src.signals.bm25 import BM25Signal, _tokenize

    rng = random.Random(seed)

    # Fit a corpus-wide BM25 once for mining.
    passage_ids = list(subset.corpus.keys())
    passage_texts = [subset.corpus[pid] for pid in passage_ids]
    bm25 = BM25Signal().fit(passage_texts)
    pid_to_idx = {pid: i for i, pid in enumerate(passage_ids)}

    # For large corpora, per-query scoring dominates wall-clock. Use an
    # inverted-index BM25 from ``rank_bm25`` when available; fall back to
    # the naive pair-scorer loop otherwise (slow but keeps smoke tests
    # self-contained).
    bm25_bulk = None
    try:
        from rank_bm25 import BM25Okapi  # type: ignore

        tokenised_corpus = [_tokenize(t) for t in passage_texts]
        bm25_bulk = BM25Okapi(tokenised_corpus, k1=1.5, b=0.75)
    except ImportError:
        pass

    pairs: list[QueryPassagePair] = []
    graded: list[float] = []
    qids: list[str] = []
    cand_k: list[int] = []

    for qid, qtext in subset.queries.items():
        positives = subset.qrels.get(qid, {})
        if not positives:
            continue

        if bm25_bulk is not None:
            scored = bm25_bulk.get_scores(_tokenize(qtext))
        else:
            scored = np.zeros(len(passage_ids), dtype=np.float64)
            for i, ptext in enumerate(passage_texts):
                scored[i] = bm25.score_pairs(
                    [QueryPassagePair(qid, passage_ids[i], qtext, ptext)]
                )[0]
        order = np.argsort(-scored, kind="mergesort")

        chosen_pids: list[str] = []
        # Always include labelled positives so the positive rate is > 0.
        for pid in positives.keys():
            if pid in pid_to_idx:
                chosen_pids.append(pid)

        # Fill up with BM25 hard negatives that are not already labelled
        # positive for this query.
        neg_budget = max_negatives_per_query
        for idx in order:
            if neg_budget <= 0:
                break
            pid = passage_ids[idx]
            if pid in positives:
                continue
            chosen_pids.append(pid)
            neg_budget -= 1

        # Shuffle so positives are not always first (prevents trivial
        # leakage via row-order in learned methods).
        rng.shuffle(chosen_pids)

        for pid in chosen_pids:
            pairs.append(
                QueryPassagePair(
                    query_id=qid,
                    passage_id=pid,
                    query_text=qtext,
                    passage_text=subset.corpus.get(pid, ""),
                )
            )
            graded.append(float(positives.get(pid, 0)))
            qids.append(qid)
        cand_k.append(len(chosen_pids))

    return pairs, graded, qids, cand_k


# ---------------------------------------------------------------------------
# query-level splits
# ---------------------------------------------------------------------------
def query_level_splits(
    qids: Iterable[str],
    ratios: tuple[float, float, float] = (0.5, 0.2, 0.3),
    seed: int = 2026,
) -> dict[str, str]:
    """50/20/30 calibration/validation/test by default, following the
    protocol fixed in the Phase-1 spec. Must be deterministic given
    the seed --- the paper reports numbers from ``seed=2026``.
    """
    uniq = sorted(set(qids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_cal = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    out: dict[str, str] = {}
    for i, qid in enumerate(uniq):
        if i < n_cal:
            out[qid] = "calibration"
        elif i < n_cal + n_val:
            out[qid] = "validation"
        else:
            out[qid] = "test"
    return out
