# CalFuse: Calibration-aware fusion of heterogeneous retrieval signals

Companion code for *Closing the Marginal–Subgroup Calibration Gap in Retrieval Fusion*.

CalFuse fuses several calibrated retrieval signals into a single probability of relevance with **multicalibration guarantees over signal-dominance subgroups**, instantiated via per-stratum Inductive Venn–Abers over a parametric base predictor (CalFuse-P). On seven BEIR subsets, CalFuse attains the lowest worst-subgroup ECE-15 on every subset; aggregate ECE understates per-subgroup miscalibration by 1.4×–7.7×.

## Repository layout

```
src/                      Library
  fusion/                   CalFuse fusion rules (parametric, copula, multical, conformal)
                            and baselines (RRF, linear-learned, single-signal, reranker)
  calibrators/              Platt, Isotonic, Temperature, learned calibrators
  signals/                  Signal computation (BM25, dense BGE/E5, cross-encoder, MonoT5,
                            PPR-graph, MinHash-LSH)
  conformal/                Inductive Venn–Abers (exact + fast PAV), Mondrian conformal,
                            sequential e-process, submodular ordering, risk control
  diagnostics/              Calibration drift, fusion mismatch, signal dependence tests
  evaluate.py               Common evaluation utilities

theory/                   LaTeX proofs for Theorems 1–4
paper/final_figures/      Canonical figures referenced by the paper, as vector PDFs.
                          Regenerable from cached eval/ artefacts via
                          scripts/make_paper_figures.py.

eval/                     Cached experimental artefacts (signal-score npz files,
                          per-method JSON results, multi-seed LLM verdicts).
                          Loader helpers: beir_loader.py, build_benchmark.py.

scripts/                  Pipelines and analyses (see "Reproducing the paper" below)
```

## Installation

Python ≥ 3.10. Core dependencies are NumPy / SciPy / scikit-learn. Heavy retrievers (cross-encoder, BGE/E5, MonoT5, LLM hallucination eval) use Torch + Transformers and are pulled in by the `full` extra.

```bash
pip install -e .              # core (CalFuse + calibrators + diagnostics)
pip install -e ".[full]"      # add Torch, transformers, BEIR loaders, matplotlib
pip install -e ".[dev]"       # pytest, ruff
```

## Smoke tests

```bash
python -m pytest scripts/smoke_test.py scripts/smoke_test_conformal.py \
                 scripts/smoke_test_multicalibration.py scripts/smoke_test_dependence.py
```

Each smoke test runs in seconds and exercises one module on synthetic data.

## Reproducing the paper

The pipeline has three stages: (1) signal computation, (2) per-method evaluation, (3) figures and tables.

### 1. Signals → cached `.npz`

For each BEIR subset, compute the signal-score matrix used by every fusion method.

```bash
python scripts/evaluate_beir.py --subset nfcorpus --out eval/beir_nfcorpus_results.npz
# repeat for: scifact, fiqa, arguana, scidocs, trec-covid, touche-2020
```

The seven `.npz` files are checked into `eval/`, so the downstream scripts run end-to-end **without re-running the dense retrievers**.

### 2. Per-method evaluation

```bash
# Worst-subgroup ECE, NDCG, calibrator + cal-size ablations:
python scripts/diagnostics_on_beir.py --subset nfcorpus
python scripts/ablate_cal_size.py --subset nfcorpus
python scripts/ablate_calibrator.py --subset nfcorpus
python scripts/conformal_on_beir.py --subset nfcorpus

# Selective NDCG / recall / prediction:
python scripts/selective_ndcg.py
python scripts/selective_recall.py
python scripts/selective_prediction.py

# Significance + signal ablation:
python scripts/significance_tests.py
python scripts/signal_ablation.py

# Drift, copula, dependence diagnostics:
python scripts/cross_dataset_drift.py
python scripts/copula_regime_simulation.py
python scripts/ablate_dependence_threshold.py
```

### 3. LLM hallucination (Section 6.4)

GPU required (A10 is sufficient). Five seeds × five BEIR subsets × Qwen-2.5-7B-Instruct:

```bash
python scripts/multiseed_llm_hallu.py --seeds 2026 2027 2028 2029 2030 \
                                      --subsets nfcorpus scifact fiqa arguana scidocs
```

Per-seed verdicts land in `eval/multiseed/llm_hallu_<subset>_seed<seed>.json`.

### 4. Selective abstention + mechanism diagnostics

```bash
python scripts/abstention_and_rank_analysis.py
python scripts/topk_composition.py
python scripts/figdata_reliability_and_ranks.py
python scripts/ivap_downstream_eval.py
```

### 5. Figures

```bash
python scripts/make_paper_figures.py
```

Writes eight figures (`fig1_worstsg_ece` through `fig8_ivap_downstream`) as both PDF and PNG into `paper/final_figures/`.

## Key methods

| Module | Purpose |
| --- | --- |
| `src.fusion.calfuse.CalFuseFusion` | CalFuse-P: parametric Bayes-optimal fusion under conditional independence. |
| `src.fusion.multicalibration.Multicalibration` | HKRR-style additive logit corrections over signal-dominance subgroups. |
| `src.fusion.calfuse_conformal.ConformalCalFuse` | Mondrian–Venn-Abers over CalFuse-P; per-stratum coverage envelope. |
| `src.fusion.calfuse_copula.CopulaCalFuse` | Closed-form Gaussian-copula fusion of calibrated logits. |
| `src.conformal.venn_abers.FastVennAbersPredictor` | PAV-based inductive Venn–Abers; returns `(p_lo, p_hi)` envelopes. |
| `src.conformal.sequential` | Anytime-valid e-process for sequential calibration monitoring. |

## License

Apache License 2.0 — see `LICENSE` for full terms and `NOTICE` for attribution requirements.

## Acknowledgments

This research was supported by [Phyvant](https://phyvant.com).

## Citation

```bibtex
@inproceedings{gunukula2026calfuse,
  title     = {Closing the Marginal--Subgroup Calibration Gap in Retrieval Fusion},
  author    = {Gunukula, Nihal},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```
