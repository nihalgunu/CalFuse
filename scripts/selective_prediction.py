"""Selective-prediction / abstention curves on cached BEIR signal matrices.

Produces a per-method coverage-vs-selective-accuracy curve for each
fusion method. This is the experimental support for the "downstream
decisions" motivation in the paper's intro: under each fusion method,
how high does selective accuracy go as we abstain on the bottom-confidence
fraction?

We use the standard one-sided rule: accept iff ``max(p, 1-p) >= tau``;
selective accuracy is 0/1 accuracy on the accepted subset; coverage is
the accepted fraction.
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

from src.evaluate import selective_accuracy_curve  # noqa: E402
from scripts.eval_from_npz import build_methods  # noqa: E402

SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


def run(npz_path: Path, methods_set):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    qids = list(data["qids"])
    split = np.asarray(data["split"])

    cal = split == "calibration"
    test = split == "test"
    qids_cal = [qids[i] for i in range(len(qids)) if cal[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test[i]]

    signal_cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    methods = build_methods(signal_cols)

    out = {}
    for name, fusion in methods.items():
        if methods_set and name not in methods_set:
            continue
        try:
            fusion.fit(X[cal], y[cal], query_ids=qids_cal)
            p = fusion.fuse(X[test], query_ids=qids_test)
            curve = selective_accuracy_curve(p, y[test])
            pts = [(pt.coverage, pt.selective_accuracy) for pt in curve.points]
            # Sample a small set of canonical operating points.
            keys = [0.50, 0.70, 0.80, 0.90, 1.00]
            samples = {f"acc@cov{k:.2f}": float(curve.at_coverage(k)) for k in keys}
            # AUC under the selective curve (trapezoid).
            xs = np.array([pt[0] for pt in pts])
            ys = np.array([pt[1] for pt in pts])
            order = np.argsort(xs)
            auc = float(np.trapz(ys[order], xs[order])) if len(pts) > 1 else float("nan")
            out[name] = {"auc": auc, **samples, "n_points": len(pts)}
        except Exception as e:
            out[name] = {"error": repr(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--out", default="eval/selective_prediction.json")
    args = ap.parse_args()

    methods_set = set(args.methods) if args.methods else None

    aggregate = {}
    for path in args.npz:
        name = Path(path).stem.replace("beir_", "").replace("_results", "")
        print(f"\n=== {name} ===")
        res = run(Path(path), methods_set)
        aggregate[name] = res
        print(f"{'method':<24} {'auc':>6} {'@.50':>6} {'@.70':>6} {'@.80':>6} {'@.90':>6} {'@1.0':>6}")
        for m, ev in res.items():
            if "error" in ev:
                print(f"{m:<24} ERROR")
                continue
            print(f"{m:<24} {ev['auc']:>6.3f} {ev['acc@cov0.50']:>6.3f} "
                  f"{ev['acc@cov0.70']:>6.3f} {ev['acc@cov0.80']:>6.3f} "
                  f"{ev['acc@cov0.90']:>6.3f} {ev['acc@cov1.00']:>6.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
