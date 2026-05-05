"""Controlled simulation showing the regime where Copula-CalFuse wins.

The headline novelty of Copula-CalFuse is closed-form fusion under
class-conditional dependence (Theorem 3). Real BEIR subsets in our
range have small label-class covariance differences and Copula loses
to Parametric on every one. This simulation quantifies *where the
crossover happens*.

We sweep the off-diagonal mass of the class-conditional covariance
difference Sigma^(1) - Sigma^(0) between strict CI (mass = 0, where
Parametric is Bayes-optimal) and a strongly-dependent regime (mass
~ 0.6, where the parametric form has visible bias by Proposition 2).
For each setting we report:

* ECE-15, NLL of Parametric vs Copula vs MLP on the test fold
* The first-order bias prediction from Prop 2 vs measured bias

This experiment supports the §Discussion claim that Parametric is
the right default but that Copula is the right tool when the
dependence diagnostic fires.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluate import evaluate
from src.fusion.calfuse import CalFuseFusion
from src.fusion.calfuse_copula import CopulaCalFuse
from src.fusion.linear_learned import LinearLearnedFusion


def make_data(n, d, rho_pos, rho_neg, base_rate, seed):
    """Generate synthetic class-conditional Gaussian-copula data.

    Each signal raw score is itself the calibrated logit (so per-signal
    Platt is a no-op and the dependence structure dominates the
    fusion comparison). We pick mean offsets so per-signal Bayes ECE
    is non-trivial.
    """
    rng = np.random.default_rng(seed)
    y = (rng.uniform(0, 1, size=n) < base_rate).astype(np.int64)
    # Class-conditional means: distinct so each signal carries info.
    mu_pos = np.array([0.6 + 0.1 * j for j in range(d)])
    mu_neg = -mu_pos
    # Equicorrelated covariance with off-diagonal rho per class.
    def cov(rho):
        return (1 - rho) * np.eye(d) + rho * np.ones((d, d))
    Cp = cov(rho_pos)
    Cn = cov(rho_neg)
    Lp = np.linalg.cholesky(Cp)
    Ln = np.linalg.cholesky(Cn)
    X = np.empty((n, d), dtype=np.float64)
    z = rng.standard_normal((n, d))
    pos = y == 1
    neg = ~pos
    X[pos] = mu_pos + z[pos] @ Lp.T
    X[neg] = mu_neg + z[neg] @ Ln.T
    qids = [f"q{i}" for i in range(n)]
    return X, y, qids


def run_one(rho_pos, rho_neg, n_cal=4000, n_test=8000, d=4, base_rate=0.3, seed=0):
    Xc, yc, qc = make_data(n_cal, d, rho_pos, rho_neg, base_rate, seed)
    Xt, yt, qt = make_data(n_test, d, rho_pos, rho_neg, base_rate, seed + 1)

    out = {}
    for name, fusion in [
        ("calfuse_parametric", CalFuseFusion(force_mode="parametric")),
        ("calfuse_copula", CopulaCalFuse(shrinkage=0.05)),
        ("linear_learned", LinearLearnedFusion()),
    ]:
        fusion.fit(Xc, yc, query_ids=qc)
        p = fusion.fuse(Xt, query_ids=qt)
        ev = evaluate(p, yt, include_reliability=False).as_dict()
        out[name] = {"ece_15": ev["ece_15"], "brier": ev["brier"], "nll": ev["nll"]}
    # First-order bias prediction (Prop 2): half-sum of off-diagonal
    # entries of Sigma^(1) - Sigma^(0). For our equicorrelated form this
    # reduces to (rho_pos - rho_neg) * d * (d - 1) / 2.
    out["predicted_bias_logodds"] = float((rho_pos - rho_neg) * d * (d - 1) / 2.0)
    out["rho_pos"] = float(rho_pos)
    out["rho_neg"] = float(rho_neg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/copula_regime.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    grid = []
    # Sweep: rho_pos increases, rho_neg fixed at 0 to maximise the off-diagonal gap.
    for rho_pos in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        per_seed = [run_one(rho_pos, 0.0, seed=s) for s in args.seeds]
        agg = {"rho_pos": rho_pos, "rho_neg": 0.0}
        for m in ["calfuse_parametric", "calfuse_copula", "linear_learned"]:
            for k in ["ece_15", "brier", "nll"]:
                vals = [r[m][k] for r in per_seed]
                agg[f"{m}_{k}_mean"] = float(np.mean(vals))
                agg[f"{m}_{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg["predicted_bias_logodds"] = per_seed[0]["predicted_bias_logodds"]
        grid.append(agg)
        print(f"rho_pos={rho_pos:.2f}  param ECE={agg['calfuse_parametric_ece_15_mean']:.4f}±"
              f"{agg['calfuse_parametric_ece_15_std']:.4f}  "
              f"copula ECE={agg['calfuse_copula_ece_15_mean']:.4f}±"
              f"{agg['calfuse_copula_ece_15_std']:.4f}  "
              f"learned ECE={agg['linear_learned_ece_15_mean']:.4f}±"
              f"{agg['linear_learned_ece_15_std']:.4f}  "
              f"pred_bias={agg['predicted_bias_logodds']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"grid": grid, "seeds": args.seeds}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
