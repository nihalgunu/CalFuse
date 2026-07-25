"""Analyze GPU-phase rebuttal results (branch rebuttal-gpu-results).

G1  Hallucination-among-non-refused at full coverage with ALL available seeds
    (10 after Phase 1), paired tests vs Linear-Learned per subset.
G2  Qwen-2.5-14B model transfer (Phase 2): same metric at 14B on
    scifact/fiqa/nfcorpus, paired tests, and 7B-vs-14B direction check.

Usage:  PYTHONPATH=. python3 rebuttal/analyze_gpu_results.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rebuttal.run_rebuttal_experiments import paired_t  # noqa: E402

OUT = REPO / "rebuttal" / "evidence"
METHODS = ["bm25_platt", "bge_platt", "linear_learned",
           "calfuse_parametric", "calfuse_conformal"]


def hallu_nr(r):
    answered = r["grounded"] + r["on_retrieval_only"] + r["fabricated"]
    return 100.0 * (r["on_retrieval_only"] + r["fabricated"]) / answered if answered else np.nan


def collect(dirpath):
    """{subset: {method: {seed: hallu_nr}}} from a verdict directory."""
    out = {}
    for f in sorted(glob.glob(str(dirpath / "llm_hallu_*_seed*.json"))):
        m = re.match(r"llm_hallu_(.+)_seed(\d+)\.json", Path(f).name)
        subset, seed = m.group(1), int(m.group(2))
        d = json.load(open(f))
        for meth, r in d["methods"].items():
            out.setdefault(subset, {}).setdefault(meth, {})[seed] = hallu_nr(r)
    return out


def tests_for(per_subset):
    res = {}
    for subset, by_m in sorted(per_subset.items()):
        seeds = sorted(set.intersection(*(set(v) for v in by_m.values())))
        entry = {m: dict(mean=float(np.mean([by_m[m][s] for s in seeds])),
                         sd=float(np.std([by_m[m][s] for s in seeds], ddof=1)),
                         per_seed={str(s): by_m[m][s] for s in seeds})
                 for m in by_m}
        t = {}
        for cf in ("calfuse_parametric", "calfuse_conformal"):
            if cf in by_m and "linear_learned" in by_m:
                t[f"{cf}__vs__linear_learned"] = paired_t(
                    [by_m[cf][s] for s in seeds],
                    [by_m["linear_learned"][s] for s in seeds])
        res[subset] = dict(n_seeds=len(seeds), seeds=seeds,
                           hallu_nr=entry, paired_tests=t)
    return res


def main():
    g1 = tests_for(collect(REPO / "eval" / "multiseed"))
    json.dump(g1, open(OUT / "g1_hallu_10seed.json", "w"), indent=1)
    print("== G1: Qwen-7B, all available seeds ==")
    for s, r in g1.items():
        for k, t in r["paired_tests"].items():
            print(f"  {s:10s} n={r['n_seeds']:2d} {k:44s} "
                  f"diff={t['mean_diff']:+6.2f}pp p={t['p']:.4f} d={t['d']:+.2f}")

    d14 = REPO / "eval" / "multiseed_qwen14b"
    if d14.exists():
        g2 = tests_for(collect(d14))
        # Direction agreement 7B vs 14B for CalFuse-P vs LL.
        agree = {}
        for s in g2:
            if s in g1:
                d7 = g1[s]["paired_tests"]["calfuse_parametric__vs__linear_learned"]["mean_diff"]
                d14b = g2[s]["paired_tests"]["calfuse_parametric__vs__linear_learned"]["mean_diff"]
                agree[s] = dict(diff_7b=d7, diff_14b=d14b,
                                same_direction=bool(np.sign(d7) == np.sign(d14b)))
        g2["_direction_agreement_calfuseP_vs_linear"] = agree
        json.dump(g2, open(OUT / "g2_qwen14b_transfer.json", "w"), indent=1)
        print("\n== G2: Qwen-14B transfer ==")
        for s, r in g2.items():
            if s.startswith("_"):
                continue
            means = {m: r["hallu_nr"][m]["mean"] for m in METHODS if m in r["hallu_nr"]}
            print(f"  {s}: " + "  ".join(f"{m}={v:.1f}" for m, v in means.items()))
            for k, t in r["paired_tests"].items():
                print(f"      {k:44s} diff={t['mean_diff']:+6.2f}pp p={t['p']:.4f} d={t['d']:+.2f}")
        print("  direction agreement:", json.dumps(agree))


if __name__ == "__main__":
    main()
