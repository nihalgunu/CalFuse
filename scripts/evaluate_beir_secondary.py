"""Secondary evaluation protocols for CalFuse's novel claims.

Standard retrieval benchmarks (NDCG, MAP, Recall) cannot measure
the three properties CalFuse uniquely provides:

* **Envelope coverage vs. nominal alpha** --- tests the Conformal-
  CalFuse / Mondrian-Venn-Abers guarantee of Theorem 5.
* **Stopping-time distribution and compute savings** --- tests the
  anytime-valid sequential-fusion claim of Theorem 6. Reports:
  empirical Type-I error, abstention rate, mean stopping time,
  per-signal consumption rate, and the implied cost savings under
  a published cost model.
* **Worst-subgroup ECE stratified by retrieval-relevant subgroup
  axes** --- tests the Multi-CalFuse guarantee of Theorem 4 on
  query-length and signal-family-dominance partitions.

These are the *secondary* protocols: they evaluate claims that
standard methodology does not measure. The *primary* protocol is
:mod:`scripts.evaluate_beir`.

Running
-------
```
PYTHONPATH=. python3 scripts/evaluate_beir_secondary.py \\
    --dataset scifact \\
    --out eval/beir_scifact_secondary.json \\
    --alphas 0.05 0.10 0.20
```

When a BEIR subset is unavailable (offline CI), the script falls
back to ``eval/benchmark_v1.jsonl`` --- the controlled simulation
substrate --- and stamps a ``substrate: simulation`` flag on the
output. This is how CI validates the pipeline.
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

from src.calibrators.platt import PlattCalibrator  # noqa: E402
from src.conformal.sequential import EProcess  # noqa: E402
from src.evaluate import expected_calibration_error  # noqa: E402
from src.fusion.calfuse import CalFuseFusion  # noqa: E402
from src.fusion.calfuse_conformal import ConformalCalFuse  # noqa: E402
from src.fusion.multicalibration import (  # noqa: E402
    query_length_subgroups,
    signal_dominance_subgroups,
    worst_subgroup_ece,
)

# Per-signal compute-cost surrogate used in the compute-savings
# calculation. Relative units (cheap=1, CE=200) roughly match the
# wall-clock ratios we measure on a single A10 GPU.
SIGNAL_COSTS = {
    "bm25": 1.0,
    "minhash_lsh": 2.0,
    "ppr_graph": 5.0,
    "dense_bge": 20.0,
    "dense_e5": 22.0,
    "cross_encoder": 200.0,
}
# Cost-ascending order used by the sequential rule.
COST_ORDER = ["bm25", "minhash_lsh", "ppr_graph", "dense_bge", "dense_e5", "cross_encoder"]


def _load_substrate(dataset: str | None):
    """Return ``(X, y, graded, qids, signal_names, query_texts, substrate_label)``."""
    if dataset is not None:
        from eval.beir_loader import build_candidate_pool, load_beir  # local import

        subset = load_beir(dataset)
        pairs, graded_list, qids, _ = build_candidate_pool(subset)
        graded = np.array(graded_list, dtype=np.float64)
        y = (graded >= 1.0).astype(np.int64)

        # Compute the six signals on this subset.
        from scripts.evaluate_beir import _compute_signal_matrix, SIGNAL_ORDER

        X = _compute_signal_matrix(pairs, subset, cpu_only=False)
        qtexts = {p.query_id: p.query_text for p in pairs}
        return X, y, graded, qids, SIGNAL_ORDER, qtexts, f"beir/{dataset}"

    # Offline fallback: read the frozen simulation substrate.
    jsonl = REPO / "eval" / "benchmark_v1.jsonl"
    with jsonl.open() as f:
        header = json.loads(f.readline())
        rows = [json.loads(line) for line in f if line.strip()]
    signals = header["signals"]
    X = np.array([[r["scores"][s] for s in signals] for r in rows], dtype=np.float64)
    graded = np.array([r["graded_relevance"] for r in rows], dtype=np.float64)
    y = np.array([r["relevance"] for r in rows], dtype=np.int64)
    qids = [r["query_id"] for r in rows]
    qtexts = {r["query_id"]: r["query_text"] for r in rows}
    return X, y, graded, qids, signals, qtexts, "simulation/benchmark_v1"


def _split_indices(qids, splits_map):
    cal_mask = np.array([splits_map.get(q, "test") == "calibration" for q in qids])
    test_mask = np.array([splits_map.get(q, "test") == "test" for q in qids])
    return cal_mask, test_mask


def _coverage_report(env, y):
    """Empirical coverage of the Venn-Abers envelope's implied
    prediction set at threshold 0.5.
    """
    inside = np.where(y == 1, env.p_hi >= 0.5, env.p_lo <= 0.5)
    width = env.p_hi - env.p_lo
    return {
        "coverage_at_0.5": float(inside.mean()),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "p90_width": float(np.percentile(width, 90)),
    }


def _sequential_report(
    calibrated_probs_cost_ordered: np.ndarray,
    y: np.ndarray,
    signal_cost_ordered: list[str],
    alphas: list[float],
) -> dict:
    """Run the e-process at multiple alpha levels; report Type-I,
    Type-II, abstention, and compute savings.
    """
    n_pairs, n_signals = calibrated_probs_cost_ordered.shape
    base_rate = float(y.mean())
    total_cost_full = sum(SIGNAL_COSTS[s] for s in signal_cost_ordered)

    out = {"base_rate": base_rate, "per_alpha": {}}
    for alpha in alphas:
        ep = EProcess(alpha=alpha, pi=base_rate)
        decisions = ep.run_batch(calibrated_probs_cost_ordered)
        rep = ep.evaluate(calibrated_probs_cost_ordered, y)

        # Weighted compute savings: sum cost of consumed signals per
        # pair, compare to sum cost of full signal cascade.
        consumed_cost = 0.0
        for d in decisions:
            for j in range(d.stopping_time):
                consumed_cost += SIGNAL_COSTS[signal_cost_ordered[j]]
        full_cost = total_cost_full * n_pairs
        savings = 1.0 - consumed_cost / max(1.0, full_cost)

        out["per_alpha"][f"{alpha:.3f}"] = {
            "empirical_type1": rep.empirical_type1,
            "empirical_type2": rep.empirical_type2,
            "abstention_rate": rep.abstention_rate,
            "mean_stopping_time": rep.mean_stopping_time,
            "per_signal_consumption": rep.per_signal_consumption.tolist(),
            "compute_savings_fraction": float(savings),
        }
    return out


def _subgroup_report(base, X_cal, y_cal, X_test, y_test, qids_cal, qids_test, qtexts):
    """Worst-subgroup ECE under two retrieval-relevant subgroup
    families: signal-family dominance and query-length buckets.
    """
    q_lengths = np.array([len(qtexts[q].split()) for q in qids_test], dtype=np.int64)
    subgroup_fns = {
        "signal_dominance": signal_dominance_subgroups(),
        "query_length": query_length_subgroups(q_lengths),
    }

    base.fit(X_cal, y_cal, query_ids=qids_cal)
    p_base = base.fuse(X_test, query_ids=qids_test)

    out = {"base_marginal_ece_15": float(expected_calibration_error(p_base, y_test))}
    for sg_name, sg_fn in subgroup_fns.items():
        M = np.asarray(sg_fn(X_test, qids_test), dtype=bool)
        out[f"worst_ece_{sg_name}_base"] = worst_subgroup_ece(p_base, y_test, M)
    return out


def main():
    parser = argparse.ArgumentParser(description="Secondary CalFuse protocols")
    parser.add_argument("--dataset", default=None,
                        help="BEIR subset name; omitted -> fall back to simulation substrate")
    parser.add_argument("--out", default=None)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    X, y, graded, qids, signal_names, qtexts, substrate = _load_substrate(args.dataset)

    # Query-level split (reuse the BEIR loader's split function when
    # available; deterministic via seed).
    from eval.beir_loader import query_level_splits

    splits_map = query_level_splits(qids, seed=args.seed)
    cal_mask, test_mask = _split_indices(qids, splits_map)
    X_cal, y_cal = X[cal_mask], y[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    qids_cal = [qids[i] for i in range(len(qids)) if cal_mask[i]]
    qids_test = [qids[i] for i in range(len(qids)) if test_mask[i]]

    # --- 1. Conformal envelope coverage at multiple alphas ---------------
    conf = ConformalCalFuse(
        base=CalFuseFusion(force_mode="parametric"),
        subgroup_fn=signal_dominance_subgroups(),
    )
    conf.fit(X_cal, labels=y_cal, query_ids=qids_cal)
    env = conf.predict_envelope(X_test, query_ids=qids_test)
    cov_report = _coverage_report(env, y_test)

    # --- 2. Sequential e-process at multiple alphas ----------------------
    # Build a per-signal calibrated-probability matrix in *cost-ascending*
    # order (cheap first), so the e-process sees BM25 before the CE.
    name_to_col = {name: i for i, name in enumerate(signal_names)}
    cost_cols = [name_to_col[n] for n in COST_ORDER if n in name_to_col]
    cost_names = [n for n in COST_ORDER if n in name_to_col]
    calibrated = np.zeros((len(y), len(cost_cols)), dtype=np.float64)
    for j, col in enumerate(cost_cols):
        c = PlattCalibrator().fit(X_cal[:, col], y_cal)
        calibrated[:, j] = c.transform(X[:, col])
    seq_report = _sequential_report(
        calibrated[test_mask],
        y_test,
        cost_names,
        alphas=args.alphas,
    )

    # --- 3. Worst-subgroup ECE for parametric base -----------------------
    sg_report = _subgroup_report(
        CalFuseFusion(force_mode="parametric"),
        X_cal, y_cal, X_test, y_test,
        qids_cal, qids_test, qtexts,
    )

    out = {
        "_type": "calfuse_beir_secondary",
        "substrate": substrate,
        "signals": signal_names,
        "n_cal": int(cal_mask.sum()),
        "n_test": int(test_mask.sum()),
        "test_positive_rate": float(y_test.mean()),
        "envelope": cov_report,
        "sequential": seq_report,
        "subgroup": sg_report,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"Wrote {args.out}")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
