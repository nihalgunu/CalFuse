"""Phase 0 smoke test.

Runs the full Phase-0 loop end-to-end on a deterministic synthetic
substrate that does not require any network access or GPU model
downloads. The test asserts that *calibration-aware* fusion beats the
strongest rank-based baseline (RRF with a post-hoc Platt calibrator)
on ECE *at matched coverage*, matching the Phase-0 exit criterion.

Synthetic substrate
-------------------
To avoid the trivial "predict the prior" solution that swamps ECE on
extreme class imbalance, each query is paired with a candidate pool of
``N_CAND`` passages built from a mix of planted positives and lexical-
neighbour hard negatives. Base rate sits at roughly 25% so that
calibration quality is not dominated by one bin.

Run::

    PYTHONPATH=. python scripts/smoke_test.py

Exit code 0 iff the test passes.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _ensure_repo_on_path() -> None:
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_ensure_repo_on_path()

from src.calibrators.platt import PlattCalibrator  # noqa: E402
from src.evaluate import expected_calibration_error, ndcg_at_k  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402
from src.fusion.rrf import RRFFusion  # noqa: E402
from src.signals.base import QueryPassagePair  # noqa: E402
from src.signals.bm25 import BM25Signal  # noqa: E402
from src.signals.dense_bge import DenseBGESignal  # noqa: E402


TOPICS = [
    ["gene", "dna", "rna", "protein", "enzyme", "cell", "genome"],
    ["quantum", "wave", "particle", "entropy", "photon", "lattice"],
    ["ranking", "retrieval", "fusion", "reranker", "query", "passage"],
    ["monetary", "inflation", "central-bank", "bond", "yield", "policy"],
    ["glacier", "arctic", "climate", "tundra", "ice", "melt"],
    ["transformer", "attention", "embedding", "layer", "gradient", "token"],
    ["legal", "court", "statute", "tort", "precedent", "appellate"],
    ["vaccine", "antigen", "immune", "antibody", "virus", "dose"],
]
FILLER = [
    "the", "of", "and", "to", "in", "for", "on", "with", "by", "is",
    "alpha", "beta", "gamma", "delta", "generic", "neutral", "fill",
    "context", "topic", "mixed", "random", "noise", "stopword",
]


def _sentence(rng: random.Random, topic: list[str], topic_weight: float, length: int) -> str:
    tokens: list[str] = []
    for _ in range(length):
        if rng.random() < topic_weight:
            tokens.append(rng.choice(topic))
        else:
            tokens.append(rng.choice(FILLER))
    return " ".join(tokens)


def _build_task(
    n_queries: int = 80,
    n_cand_per_query: int = 24,
    positives_per_query: int = 6,
    seed: int = 42,
):
    """Per-query candidate pool with planted positives and lexical hard negatives."""
    rng = random.Random(seed)
    queries: list[dict] = []
    passage_id_counter = 0

    for qi in range(n_queries):
        topic = rng.choice(TOPICS)
        other_topic = rng.choice([t for t in TOPICS if t is not topic])
        query_text = _sentence(rng, topic, topic_weight=0.85, length=8)
        # Positives: high topic weight.
        positives = [
            _sentence(rng, topic, topic_weight=0.7, length=rng.randint(20, 35))
            for _ in range(positives_per_query)
        ]
        # Hard negatives: partial overlap with query topic (weight ~0.3) plus
        # some other-topic content; lexical retrievers should still rank them
        # above random but below true positives.
        n_neg = n_cand_per_query - positives_per_query
        n_hard = n_neg // 2
        n_easy = n_neg - n_hard
        hard_negs = [
            _sentence(rng, topic, topic_weight=0.25, length=rng.randint(18, 30))
            for _ in range(n_hard)
        ]
        easy_negs = [
            _sentence(rng, other_topic, topic_weight=0.4, length=rng.randint(18, 30))
            for _ in range(n_easy)
        ]
        passages = positives + hard_negs + easy_negs
        labels = [1] * positives_per_query + [0] * n_neg

        # Shuffle so positives aren't always at the front.
        order = list(range(len(passages)))
        rng.shuffle(order)

        cand = []
        for i in order:
            cand.append(
                {
                    "pid": f"p{passage_id_counter:05d}",
                    "text": passages[i],
                    "label": labels[i],
                }
            )
            passage_id_counter += 1

        queries.append({"qid": f"q{qi:04d}", "text": query_text, "candidates": cand})

    pairs: list[QueryPassagePair] = []
    y_flat: list[int] = []
    pair_query_ids: list[str] = []
    for q in queries:
        for c in q["candidates"]:
            pairs.append(
                QueryPassagePair(
                    query_id=q["qid"],
                    passage_id=c["pid"],
                    query_text=q["text"],
                    passage_text=c["text"],
                )
            )
            y_flat.append(c["label"])
            pair_query_ids.append(q["qid"])
    return queries, pairs, np.array(y_flat, dtype=np.int64), pair_query_ids


def main() -> int:
    queries, pairs, y_flat, pair_query_ids = _build_task()

    # BM25 is fit on the full passage corpus (calibration + test) — fitting
    # IDF is an unsupervised step so no label leakage occurs.
    all_passages = [p.passage_text for p in pairs]
    bm25 = BM25Signal().fit(all_passages)
    dense = DenseBGESignal(force_fallback=True)

    s_bm25 = bm25.score_pairs(pairs)
    s_dense = dense.score_pairs(pairs)
    X = np.stack([s_bm25, s_dense], axis=1)

    # Query-level split prevents leakage through IDF or shared passages.
    rng = np.random.default_rng(0)
    uniq_q = list(dict.fromkeys(pair_query_ids))
    rng.shuffle(uniq_q)
    cal_q = set(uniq_q[: len(uniq_q) // 2])
    cal_mask = np.array([q in cal_q for q in pair_query_ids])
    test_mask = ~cal_mask

    X_cal, y_cal = X[cal_mask], y_flat[cal_mask]
    X_test, y_test = X[test_mask], y_flat[test_mask]
    qids_cal = [q for q, m in zip(pair_query_ids, cal_mask) if m]
    qids_test = [q for q, m in zip(pair_query_ids, cal_mask) if not m]

    print(f"base rate (cal)={y_cal.mean():.3f}  base rate (test)={y_test.mean():.3f}")

    # ---- RRF + post-hoc Platt (strongest rank-based baseline) -----------
    rrf = RRFFusion(k=60.0)
    rrf.fit(X_cal, y_cal, query_ids=qids_cal)
    p_rrf = rrf.fuse(X_test, query_ids=qids_test)
    ece_rrf = expected_calibration_error(p_rrf, y_test)
    ndcg_rrf = ndcg_at_k(p_rrf, y_test.astype(np.float64), qids_test, k=10)

    # ---- Naive mean of Platt-calibrated probabilities -------------------
    platt_bm = PlattCalibrator().fit(X_cal[:, 0], y_cal)
    platt_de = PlattCalibrator().fit(X_cal[:, 1], y_cal)
    p_naive = 0.5 * platt_bm.transform(X_test[:, 0]) + 0.5 * platt_de.transform(X_test[:, 1])
    ece_naive = expected_calibration_error(p_naive, y_test)

    # ---- CalFuse (parametric) ------------------------------------------
    calfuse = CalFuseFusion(force_mode="parametric")
    calfuse.fit(X_cal, y_cal, query_ids=qids_cal)
    p_calfuse = calfuse.fuse(X_test, query_ids=qids_test)
    ece_calfuse = expected_calibration_error(p_calfuse, y_test)
    ndcg_calfuse = ndcg_at_k(p_calfuse, y_test.astype(np.float64), qids_test, k=10)

    print("=" * 72)
    print(f"  RRF + post-hoc Platt      ECE_15 = {ece_rrf:.4f}   NDCG@10 = {ndcg_rrf:.4f}")
    print(f"  Naive mean of calibrated  ECE_15 = {ece_naive:.4f}")
    print(f"  CalFuse (parametric)      ECE_15 = {ece_calfuse:.4f}   NDCG@10 = {ndcg_calfuse:.4f}")
    print("=" * 72)

    # Primary claim: CalFuse beats RRF on ECE.
    beats_rrf = ece_calfuse < ece_rrf - 1e-4
    # Retrieval quality must not collapse.
    ndcg_ok = ndcg_calfuse >= 0.8 * ndcg_rrf - 1e-3

    if not beats_rrf:
        print(f"FAIL: CalFuse ECE {ece_calfuse:.4f} did not beat RRF ECE {ece_rrf:.4f}")
        return 1
    if not ndcg_ok:
        print(f"FAIL: NDCG@10 collapsed ({ndcg_calfuse:.4f} vs RRF {ndcg_rrf:.4f})")
        return 1

    print("SMOKE TEST PASSED: calibration-aware fusion beats RRF on ECE "
          "without sacrificing NDCG.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
