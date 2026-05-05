"""Build the *controlled simulation substrate* used for theorem-
targeted sanity checks.

Scope
-----
This script intentionally does **not** build a new public
benchmark. Our primary evaluation uses BEIR subsets via
:mod:`eval.beir_loader` and :mod:`scripts.evaluate_beir`, with
standard retrieval methodology (Thakur et al., 2021).

The artefact this script emits --- ``eval/benchmark_v1.jsonl`` ---
is a deterministic synthetic substrate with the same schema as our
BEIR pipeline. It exists for three purposes: (i) smoke tests,
(ii) regression tests of the full pipeline without network or GPU,
(iii) theorem-targeted simulations that instantiate exactly the
regime a specific theorem assumes (e.g. class-conditional
covariance shift for Theorem 3, subgroup miscalibration for
Theorem 4). Numbers from this substrate are not comparable to
BEIR leaderboards and are not the paper's headline numbers.

The word ``benchmark`` appears in the file name for historical
Phase-1 naming reasons; the paper's evaluation section
(Section 7) is authoritative on the role this substrate plays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.signals.base import QueryPassagePair  # noqa: E402
from src.signals.bm25 import BM25Signal  # noqa: E402
from src.signals.cross_encoder import CrossEncoderSignal  # noqa: E402
from src.signals.dense_bge import DenseBGESignal  # noqa: E402
from src.signals.dense_e5 import DenseE5Signal  # noqa: E402
from src.signals.minhash_lsh import MinHashLSHSignal  # noqa: E402
from src.signals.ppr_graph import PPRGraphSignal  # noqa: E402


SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


TIER_SIGNALS = {
    "homogeneous_lexical": ["bm25", "minhash_lsh"],
    "homogeneous_dense": ["dense_bge", "dense_e5"],
    "hetero_independent": ["bm25", "ppr_graph", "cross_encoder"],
    "hetero_dependent": ["bm25", "dense_bge", "dense_e5"],
}


# ---------------------------------------------------------------------------
# Synthetic substrate
# ---------------------------------------------------------------------------
TOPICS = [
    ["gene", "dna", "rna", "protein", "enzyme", "cell", "genome", "chromosome"],
    ["quantum", "wave", "particle", "entropy", "photon", "lattice", "spin"],
    ["ranking", "retrieval", "fusion", "reranker", "query", "passage", "index"],
    ["monetary", "inflation", "central-bank", "bond", "yield", "policy", "rate"],
    ["glacier", "arctic", "climate", "tundra", "ice", "melt", "permafrost"],
    ["transformer", "attention", "embedding", "layer", "gradient", "token"],
    ["legal", "court", "statute", "tort", "precedent", "appellate", "judgment"],
    ["vaccine", "antigen", "immune", "antibody", "virus", "dose", "titre"],
    ["network", "router", "packet", "latency", "bandwidth", "congestion"],
    ["trial", "patient", "cohort", "placebo", "blinded", "efficacy", "outcome"],
]
FILLER = [
    "the", "of", "and", "to", "in", "for", "on", "with", "by", "is",
    "alpha", "beta", "gamma", "delta", "generic", "neutral", "fill",
    "context", "topic", "mixed", "random", "noise", "stopword",
]


def _sentence(rng: random.Random, topic: list[str], topic_weight: float, length: int) -> str:
    toks = []
    for _ in range(length):
        toks.append(rng.choice(topic) if rng.random() < topic_weight else rng.choice(FILLER))
    return " ".join(toks)


@dataclass
class PairRecord:
    query_id: str
    passage_id: str
    query_text: str
    passage_text: str
    scores: dict
    relevance: int
    graded_relevance: float
    split: str  # calibration | validation | test
    substrate: str  # synthetic | beir_nfcorpus | ...


def build_synthetic(
    n_queries: int = 200,
    n_cand_per_query: int = 24,
    positives_per_query: int = 5,
    seed: int = 2026,
) -> tuple[list[dict], list[QueryPassagePair], np.ndarray, list[str]]:
    rng = random.Random(seed)
    queries: list[dict] = []
    pid_counter = 0

    for qi in range(n_queries):
        topic = rng.choice(TOPICS)
        other = rng.choice([t for t in TOPICS if t is not topic])
        query_text = _sentence(rng, topic, topic_weight=0.85, length=8)

        # Graded relevance: the ``positives`` set has label 2; a few
        # "partial" passages (weight 0.5) get label 1; hard negatives 0.
        passages = []
        labels_graded = []
        for _ in range(positives_per_query):
            passages.append(_sentence(rng, topic, 0.7, rng.randint(20, 35)))
            labels_graded.append(2.0)
        n_partial = 2
        for _ in range(n_partial):
            passages.append(_sentence(rng, topic, 0.45, rng.randint(20, 30)))
            labels_graded.append(1.0)
        n_neg = n_cand_per_query - positives_per_query - n_partial
        n_hard = n_neg // 2
        n_easy = n_neg - n_hard
        for _ in range(n_hard):
            passages.append(_sentence(rng, topic, 0.25, rng.randint(18, 30)))
            labels_graded.append(0.0)
        for _ in range(n_easy):
            passages.append(_sentence(rng, other, 0.4, rng.randint(18, 30)))
            labels_graded.append(0.0)

        order = list(range(len(passages)))
        rng.shuffle(order)
        cand = []
        for i in order:
            cand.append(
                {
                    "pid": f"p{pid_counter:06d}",
                    "text": passages[i],
                    "graded": labels_graded[i],
                    "label": int(labels_graded[i] >= 1.0),
                }
            )
            pid_counter += 1
        queries.append({"qid": f"q{qi:05d}", "text": query_text, "candidates": cand})

    pairs: list[QueryPassagePair] = []
    graded: list[float] = []
    qids: list[str] = []
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
            graded.append(c["graded"])
            qids.append(q["qid"])
    return queries, pairs, np.array(graded, dtype=np.float64), qids


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------
def compute_signals(
    pairs: Sequence[QueryPassagePair],
    passage_ids: Sequence[str],
    passage_texts: Sequence[str],
    force_offline: bool = True,
) -> np.ndarray:
    """Return ``(n_pairs, 6)`` raw-score matrix in ``SIGNAL_ORDER``."""
    all_texts = [p.passage_text for p in pairs]
    # BM25 IDF is fit on the full passage set — unsupervised, no leakage.
    bm25 = BM25Signal().fit(all_texts)
    bge = DenseBGESignal(force_fallback=force_offline)
    e5 = DenseE5Signal(force_fallback=force_offline)
    ce = CrossEncoderSignal(force_fallback=force_offline)
    ppr = PPRGraphSignal().fit_corpus(passage_ids, passage_texts)
    mh = MinHashLSHSignal()

    return np.stack(
        [
            bm25.score_pairs(pairs),
            bge.score_pairs(pairs),
            e5.score_pairs(pairs),
            ce.score_pairs(pairs),
            ppr.score_pairs(pairs),
            mh.score_pairs(pairs),
        ],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------
def query_level_splits(query_ids: Iterable[str], seed: int = 2026) -> dict[str, str]:
    uniq = sorted(set(query_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_cal = int(round(0.5 * n))
    n_val = int(round(0.2 * n))
    splits: dict[str, str] = {}
    for i, qid in enumerate(uniq):
        if i < n_cal:
            splits[qid] = "calibration"
        elif i < n_cal + n_val:
            splits[qid] = "validation"
        else:
            splits[qid] = "test"
    return splits


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def write_benchmark(
    records: list[PairRecord],
    jsonl_path: Path,
    labels_path: Path,
    tier_membership: dict[str, list[str]],
    substrate: str,
    seed: int,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Build body first so we can checksum it.
    body_lines = [json.dumps(asdict(r), sort_keys=True) for r in records]
    body_bytes = ("\n".join(body_lines) + "\n").encode("utf-8")
    checksum = hashlib.sha256(body_bytes).hexdigest()

    header = {
        "_type": "calfuse_benchmark_v1_header",
        "substrate": substrate,
        "seed": seed,
        "n_pairs": len(records),
        "signals": SIGNAL_ORDER,
        "tiers": tier_membership,
        "body_sha256": checksum,
    }

    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, sort_keys=True) + "\n")
        f.write("\n".join(body_lines) + "\n")

    labels = {
        "splits": {f"{r.query_id}:{r.passage_id}": r.split for r in records},
        "relevance": {f"{r.query_id}:{r.passage_id}": r.relevance for r in records},
        "graded_relevance": {
            f"{r.query_id}:{r.passage_id}": r.graded_relevance for r in records
        },
        "tiers": tier_membership,
    }
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, sort_keys=True)


def verify_frozen(jsonl_path: Path) -> None:
    with jsonl_path.open("r", encoding="utf-8") as f:
        header_line = f.readline()
        body_bytes = f.read().encode("utf-8")
    header = json.loads(header_line)
    expected = header["body_sha256"]
    actual = hashlib.sha256(body_bytes).hexdigest()
    if expected != actual:
        raise RuntimeError(
            f"benchmark body checksum mismatch: expected {expected}, got {actual}"
        )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Build CalFuse benchmark v1")
    parser.add_argument("--substrate", choices=["synthetic", "beir"], default="synthetic")
    parser.add_argument("--out-jsonl", default="eval/benchmark_v1.jsonl")
    parser.add_argument("--out-labels", default="eval/ground_truth_relevance.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true", help="overwrite a frozen benchmark")
    parser.add_argument(
        "--n-queries",
        type=int,
        default=200,
        help="synthetic mode only: number of queries",
    )
    args = parser.parse_args()

    jsonl_path = REPO / args.out_jsonl
    labels_path = REPO / args.out_labels

    if jsonl_path.exists() and not args.force:
        print(
            f"refusing to overwrite {jsonl_path}; pass --force to rebuild "
            "(Phase 1 protocol forbids silent mutation of the frozen benchmark)"
        )
        return 1

    if args.substrate == "synthetic":
        queries, pairs, graded, qids = build_synthetic(
            n_queries=args.n_queries, seed=args.seed
        )
        substrate_label = "synthetic_v1"
    else:
        raise NotImplementedError(
            "BEIR substrate mode requires `ir-datasets` and `beir` installed; "
            "see docstring of build_benchmark.py for the loader spec."
        )

    passage_ids = [p.passage_id for p in pairs]
    passage_texts = [p.passage_text for p in pairs]
    X = compute_signals(pairs, passage_ids, passage_texts, force_offline=True)

    splits = query_level_splits(qids, seed=args.seed)

    records: list[PairRecord] = []
    for i, p in enumerate(pairs):
        scores = {name: float(X[i, j]) for j, name in enumerate(SIGNAL_ORDER)}
        records.append(
            PairRecord(
                query_id=p.query_id,
                passage_id=p.passage_id,
                query_text=p.query_text,
                passage_text=p.passage_text,
                scores=scores,
                relevance=int(graded[i] >= 1.0),
                graded_relevance=float(graded[i]),
                split=splits[p.query_id],
                substrate=substrate_label,
            )
        )

    tier_membership = {
        tier: TIER_SIGNALS[tier] for tier in TIER_SIGNALS
    }

    write_benchmark(records, jsonl_path, labels_path, tier_membership, substrate_label, args.seed)
    verify_frozen(jsonl_path)

    # Summary.
    n_cal = sum(1 for r in records if r.split == "calibration")
    n_val = sum(1 for r in records if r.split == "validation")
    n_test = sum(1 for r in records if r.split == "test")
    print(f"wrote {jsonl_path} ({len(records)} pairs)")
    print(f"  calibration={n_cal}  validation={n_val}  test={n_test}")
    print(f"  substrate={substrate_label}  seed={args.seed}")
    print(f"  tiers: {', '.join(tier_membership.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
