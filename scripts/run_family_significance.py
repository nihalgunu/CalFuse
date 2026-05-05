"""Pairwise significance tests for the subgroup-stratified family.

For each subset and each method-pair, run a paired test across the
5 seeds of (worst-subgroup-ECE, ECE-15, NDCG@10). Reports paired
$t$-statistic, $p$-value, and Cohen's $d$ effect size.

Inputs come from `eval/subgroup_calibrator_family.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ["calfuse_parametric_subgroup_platt",
           "calfuse_parametric_subgroup_isotonic",
           "calfuse_conformal",
           "calfuse_multical"]
LABEL = {
    "calfuse_parametric_subgroup_platt": "S-Platt",
    "calfuse_parametric_subgroup_isotonic": "S-Isotonic",
    "calfuse_conformal": "Conformal",
    "calfuse_multical": "Multi-CalFuse",
}
SUBSETS = ["nfcorpus", "scifact", "fiqa", "trec-covid",
           "arguana", "scidocs", "touche-2020"]
METRICS = [("worst_subgroup_ece_15", "WSECE", "lower"),
           ("ece_15", "ECE15", "lower"),
           ("ndcg_10", "NDCG10", "higher")]


def paired_t(a, b):
    diff = a - b
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if sd == 0:
        return mean, float("inf") if mean != 0 else 0.0, 1.0 if mean == 0 else 0.0
    t = mean / (sd / np.sqrt(n))
    # Two-sided p via Student-t survival; small n=5 means rough.
    from scipy import stats  # type: ignore
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))
    cohens_d = mean / sd
    return mean, t, p, cohens_d


def fmt_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="eval/subgroup_calibrator_family.json")
    ap.add_argument("--out", default="eval/family_significance.json")
    args = ap.parse_args()

    d = json.load(open(args.src))
    out = {}
    for met_key, met_label, direction in METRICS:
        print(f"\n=== {met_label} ===")
        out[met_key] = {}
        for subset in SUBSETS:
            if subset not in d:
                continue
            per_seed = d[subset]["per_seed"]
            seeds = sorted(per_seed.keys())
            vals = {m: np.array([per_seed[s].get(m, {}).get(met_key, np.nan) for s in seeds],
                                dtype=np.float64) for m in METHODS}
            print(f"  --- {subset} ---")
            print(f"  {'pair':<28} {'mean diff':>12} {'t':>7} {'p':>8} {'sig':>4} {'d':>6}")
            out[met_key][subset] = {}
            for i, m1 in enumerate(METHODS):
                for j, m2 in enumerate(METHODS):
                    if i >= j:
                        continue
                    a = vals[m1]; b = vals[m2]
                    if np.isnan(a).any() or np.isnan(b).any():
                        continue
                    res = paired_t(a, b)
                    if len(res) == 3:
                        mean, t, p = res; cd = float("nan")
                    else:
                        mean, t, p, cd = res
                    pair = f"{LABEL[m1]} - {LABEL[m2]}"
                    sig = fmt_p(p)
                    print(f"  {pair:<28} {mean:>+12.4f} {t:>+7.2f} {p:>8.4f} {sig:>4} {cd:>+6.2f}")
                    out[met_key][subset][f"{m1}__vs__{m2}"] = {
                        "mean_diff": mean, "t": t, "p": p, "cohens_d": cd,
                    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    print(f"\nSig codes: . p<.10  * p<.05  ** p<.01  *** p<.001")


if __name__ == "__main__":
    main()
