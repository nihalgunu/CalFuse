"""Leave-one-signal-out ablation on cached BEIR signal matrices.

For each subset, runs each fusion method with each signal removed in
turn. Quantifies the marginal contribution of each signal so the paper
can either justify keeping PPR/MinHash (which have been suspected to
contribute nothing) or drop them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluate import evaluate  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402
from src.fusion.linear_learned import LinearLearnedFusion  # noqa: E402
from src.fusion.multicalibration import (  # noqa: E402
    Multicalibration,
    signal_dominance_subgroups,
    worst_subgroup_ece,
)
from src.fusion.rrf import RRFFusion  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def make_methods():
    return {
        "rrf": lambda: RRFFusion(),
        "linear_learned": lambda: LinearLearnedFusion(),
        "calfuse_parametric": lambda: CalFuseFusion(force_mode="parametric"),
        "calfuse_multical": lambda: Multicalibration(
            base=CalFuseFusion(force_mode="parametric"),
            subgroup_fn=signal_dominance_subgroups(),
        ),
    }


def run(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    graded = data["graded"].astype(np.float64)
    qids = list(data["qids"])
    split = np.asarray(data["split"])

    cal = split == "calibration"
    test = split == "test"
    qids_cal = [qids[i] for i in range(len(qids)) if cal[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test[i]]

    methods = make_methods()
    out = {"all": {}, "leave_out": {}}

    # Baseline: all signals in.
    for name, factory in methods.items():
        fusion = factory()
        fusion.fit(X[cal], y[cal], query_ids=qids_cal)
        p = fusion.fuse(X[test], query_ids=qids_test)
        ev = evaluate(p, y[test], graded_labels=graded[test],
                      query_ids=qids_test, include_reliability=False).as_dict()
        out["all"][name] = ev

    # Leave each signal out.
    for k, sig in enumerate(SIGNAL_ORDER):
        keep = [j for j in range(X.shape[1]) if j != k]
        X_lo = X[:, keep]
        out["leave_out"][sig] = {}
        for name, factory in methods.items():
            try:
                fusion = factory()
                fusion.fit(X_lo[cal], y[cal], query_ids=qids_cal)
                p = fusion.fuse(X_lo[test], query_ids=qids_test)
                ev = evaluate(p, y[test], graded_labels=graded[test],
                              query_ids=qids_test, include_reliability=False).as_dict()
                out["leave_out"][sig][name] = ev
            except Exception as e:
                out["leave_out"][sig][name] = {"error": repr(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--out", default="eval/signal_ablation.json")
    args = ap.parse_args()

    aggregate = {}
    for path in args.npz:
        name = Path(path).stem.replace("beir_", "").replace("_results", "")
        print(f"\n=== {name} ===")
        res = run(Path(path))
        aggregate[name] = res
        # Pretty print: deltas vs all-in for calfuse_parametric.
        base_ndcg = res["all"]["calfuse_parametric"].get("ndcg_10") or 0.0
        base_ece = res["all"]["calfuse_parametric"].get("ece_15") or 0.0
        print(f"calfuse_parametric (all signals)   NDCG@10={base_ndcg:.4f}  ECE15={base_ece:.4f}")
        for sig in SIGNAL_ORDER:
            ev = res["leave_out"][sig].get("calfuse_parametric", {})
            if "error" in ev:
                continue
            n = ev.get("ndcg_10") or 0.0
            e = ev.get("ece_15") or 0.0
            print(f"  drop {sig:<14}  dNDCG={n - base_ndcg:+.4f}  dECE={e - base_ece:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
