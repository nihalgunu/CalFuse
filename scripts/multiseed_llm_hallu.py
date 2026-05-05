"""Multi-seed wrapper around llm_hallucination_eval.py.

For each (subset, seed) we:
1. Re-run the LLM hallucination eval with that seed (which controls
   the random sample of 50 evaluation queries from the eligible set).
2. Aggregate verdicts across seeds: mean +/- std hallucination rate,
   grounded fraction, refusal rate.

Five seeds × five subsets × ~6-15 min each ≈ 3-4 hours of LLM time.

Output goes to eval/llm_hallu_{subset}_seed{seed}.json individually
and to eval/llm_hallu_multiseed.json as the aggregate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


SUBSETS = ["nfcorpus", "scifact", "fiqa", "arguana", "scidocs"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
METHODS = ["bm25_platt", "bge_platt", "linear_learned",
           "calfuse_parametric", "calfuse_conformal"]


def run_eval(subset, seed, model, n_queries, top_k):
    out_path = f"eval/llm_hallu_{subset}_seed{seed}.json"
    if Path(out_path).exists():
        print(f"  cache hit: {out_path}")
        return out_path
    cmd = [sys.executable, "scripts/llm_hallucination_eval.py",
           "--subset", subset, "--model", model,
           "--n-queries", str(n_queries), "--top-k", str(top_k),
           "--seed", str(seed),
           "--methods"] + METHODS + ["--out", out_path]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["PYTHONHASHSEED"] = "0"
    print(f"  running seed={seed} on {subset}...")
    subprocess.check_call(cmd, env=env)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", default="eval/llm_hallu_multiseed.json")
    args = ap.parse_args()

    results = {}
    for subset in args.subsets:
        results[subset] = {"per_seed": {}, "aggregated": {}}
        print(f"\n=== {subset} ===")
        for seed in args.seeds:
            path = run_eval(subset, seed, args.model, args.n_queries, args.top_k)
            d = json.load(open(path))
            results[subset]["per_seed"][seed] = d["methods"]

        agg = {}
        for m in METHODS:
            n_arr, hal_arr, grd_arr, ret_arr, fab_arr, ref_arr, posk_arr = (
                [], [], [], [], [], [], [])
            for seed in args.seeds:
                ev = results[subset]["per_seed"][seed].get(m, {})
                if "error" in ev or not ev: continue
                n = ev["n_queries"]
                if n == 0: continue
                n_arr.append(n)
                hal_arr.append((ev["fabricated"] + ev["on_retrieval_only"]) / n)
                grd_arr.append(ev["grounded"] / n)
                ret_arr.append(ev["on_retrieval_only"] / n)
                fab_arr.append(ev["fabricated"] / n)
                ref_arr.append(ev["refused"] / n)
                posk_arr.append(ev["fraction_pos_in_topk"])
                # Hallucination among non-refused.
                non_ref = ev["fabricated"] + ev["on_retrieval_only"] + ev["grounded"]
                if non_ref > 0:
                    pass  # store both denominators below
            if not n_arr: continue
            def m_s(a):
                arr = np.array(a, dtype=float)
                return (float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0))
            agg[m] = {
                "n_seeds": len(n_arr),
                "hallu_mean": m_s(hal_arr)[0], "hallu_std": m_s(hal_arr)[1],
                "grounded_mean": m_s(grd_arr)[0], "grounded_std": m_s(grd_arr)[1],
                "on_retrieval_mean": m_s(ret_arr)[0], "on_retrieval_std": m_s(ret_arr)[1],
                "fabricated_mean": m_s(fab_arr)[0], "fabricated_std": m_s(fab_arr)[1],
                "refused_mean": m_s(ref_arr)[0], "refused_std": m_s(ref_arr)[1],
                "pos_in_topk_mean": m_s(posk_arr)[0], "pos_in_topk_std": m_s(posk_arr)[1],
            }
        results[subset]["aggregated"] = agg

        # Print summary.
        print(f"  {'method':<22} {'hallu':>16} {'grounded':>16} {'refused':>14}")
        for m in METHODS:
            a = agg.get(m)
            if not a: continue
            print(f"  {m:<22} {a['hallu_mean']*100:>5.1f}±{a['hallu_std']*100:>5.1f}%  "
                  f"{a['grounded_mean']*100:>5.1f}±{a['grounded_std']*100:>5.1f}%  "
                  f"{a['refused_mean']*100:>5.1f}±{a['refused_std']*100:>5.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
