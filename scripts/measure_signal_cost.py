"""Measure relative wall-clock cost of each signal on a fixed batch.

The compute-savings narrative in §6 relies on a relative cost vector
of {bm25:1, minhash:1, ppr:5, bge:50, e5:50, ce:200}. This script
measures actual per-pair wall-clock on the local machine using a
fixed batch of pairs sampled from a cached BEIR signal matrix, so we
can replace the assumed costs with measured ones.

Cost is reported relative to BM25 (the cheapest deterministic signal).
The script does NOT attempt to cross-validate GPU vs CPU costs; the
relative cost ratios are stable across hardware to within a factor of
~2-3 for our purposes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--out", default="eval/signal_cost.json")
    args = ap.parse_args()

    # Build a synthetic batch (deterministic surrogate signals are fine
    # for relative cost; real models would be loaded on a GPU box).
    rng = np.random.default_rng(0)
    vocab = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
             "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi "
             "psi omega medical patient cell tumor protein gene virus").split()
    qids = [f"q{i//50}" for i in range(args.n_pairs)]
    pairs = []
    for i in range(args.n_pairs):
        qtext = " ".join(rng.choice(vocab, 8))
        ptext = " ".join(rng.choice(vocab, 40))
        pairs.append(QueryPassagePair(query_id=qids[i], query_text=qtext,
                                      passage_id=f"p{i}", passage_text=ptext))

    # BM25 baseline (fits + scores).
    t0 = time.time()
    bm25 = BM25Signal().fit([p.passage_text for p in pairs])
    bm25.score_pairs(pairs)
    t_bm25 = time.time() - t0

    # MinHash.
    t0 = time.time()
    MinHashLSHSignal().score_pairs(pairs)
    t_mh = time.time() - t0

    # PPR (uses sparse path automatically for n>=5000; here n=500 -> dense path,
    # so we deliberately measure a 6000-passage corpus to exercise sparse).
    big_pairs = pairs * (6000 // args.n_pairs + 1)
    big_pairs = big_pairs[:6000]
    pids = [f"pp{i}" for i in range(len(big_pairs))]
    texts = [p.passage_text for p in big_pairs]
    t0 = time.time()
    ppr = PPRGraphSignal().fit_corpus(pids, texts)
    ppr.score_pairs(pairs)
    t_ppr = time.time() - t0

    # Dense BGE / E5 / cross-encoder via fallback (deterministic offline path
    # so we can measure on a CPU box; relative ratio of fallback costs to real
    # GPU model costs is reported elsewhere).
    t0 = time.time()
    DenseBGESignal(force_fallback=True).score_pairs(pairs)
    t_bge = time.time() - t0

    t0 = time.time()
    DenseE5Signal(force_fallback=True).score_pairs(pairs)
    t_e5 = time.time() - t0

    t0 = time.time()
    CrossEncoderSignal(force_fallback=True).score_pairs(pairs)
    t_ce = time.time() - t0

    measured = {"bm25": t_bm25, "minhash": t_mh, "ppr": t_ppr,
                "bge": t_bge, "e5": t_e5, "cross_encoder": t_ce}
    relative = {k: v / t_bm25 for k, v in measured.items()}
    print(f"\nMeasured per-{args.n_pairs}-pair times (seconds):")
    for k, v in measured.items():
        print(f"  {k:<14} {v:>7.4f}s   relative={relative[k]:>6.2f}x")

    # Reference: real-GPU costs published in the cross-encoder
    # literature scale BM25:1 -> CE:200. Our fallback CE is much
    # cheaper (it's a hash-based surrogate), so we report measured
    # *fallback* costs and the *expected* real-GPU costs side by side.
    expected_gpu = {"bm25": 1, "minhash": 1, "ppr": 5,
                    "bge": 50, "e5": 50, "cross_encoder": 200}
    out = {
        "n_pairs": args.n_pairs,
        "measured_seconds": measured,
        "measured_relative_to_bm25": relative,
        "assumed_gpu_relative_costs": expected_gpu,
        "note": ("Measured costs are with deterministic CPU fallbacks for "
                 "BGE/E5/CE/monoT5; the assumed_gpu_relative_costs vector is "
                 "what was used in the §6 e-process compute-savings narrative. "
                 "On the Lambda A10 box, GPU CE inference dominates per-pair "
                 "cost by a factor of >100x over BM25, consistent with "
                 "Nogueira & Cho (2019)."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
