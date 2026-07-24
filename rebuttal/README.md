# Rebuttal drafts — NeurIPS 2026 submission 33259

Drafts of the author responses for the discussion phase. One file per reply:

- `00_global_response.md` — top-level comment addressing AC Sraq's meta review (jargon / self-containedness) and the cross-cutting theory-interpretation concern.
- `01_response_reviewer_utWJ.md` — borderline accept; Q1 ranking-vs-probability, Q2 subgroup circularity, Q3 Venn–Abers vs Subgroup-Platt, Q4 hallucination robustness, plus rare-strata and exchangeability weaknesses.
- `02_response_reviewer_T3MF.md` — borderline accept (champion); lower-bound interpretation, hallucination scope, selective-NDCG evidence, IVAP-envelope value.
- `03_response_reviewer_VkVU.md` — reject; novelty, direct theory for CalFuse, end-to-end impact.

## Every number traces to a released artifact

| Claim in drafts | Source |
|---|---|
| Worst-subgroup ECE-15 table, marginal-vs-worst gap (1.4–7.7×), per-subset means | `eval/multi_seed_<subset>.json` (`aggregated`) |
| NDCG unchanged vs Linear-Learned (nfcorpus p=0.91, arguana p=0.61) | `eval/significance_tests.json` |
| Subgroup-Platt vs Subgroup-Isotonic vs IVAP paired stats (fiqa p<0.001 d=3.9; scifact p=0.043 d=1.3; nfcorpus tie p=0.21) | `eval/family_significance.json`, `eval/subgroup_calibrator_family.json`, `eval/subgroup_platt_ablation.json` |
| Five-subset hallucination-among-non-refused table at full coverage | `eval/selective_abstention_multiseed.json` (coverage "1.0"), raw verdicts in `eval/multiseed/llm_hallu_*.json` |
| scifact −3.32pp, p=0.045, d=−1.29 (CalFuse-P vs Linear-Learned) | `scripts/make_paper_figures.py` (fig6), `eval/selective_abstention_multiseed.json` |
| Selective curve scifact 30.8% → 15.2% at coverage 0.5 | `eval/selective_abstention_multiseed.json`, `eval/ivap_downstream_eval.json` |
| Rank-of-first-positive unchanged (mechanism test) | `eval/mean_rank_positive_multiseed.json`, `eval/figdata_ranks_scifact.json` |
| Rare-stratum fallback (min 30 pairs + positive floor → pooled IVAP) | `src/conformal/mondrian.py` |
| Cal-size ablation (10% calibration: worst-sg ECE 0.023 vs ~0.021 full) | `eval/ablate_cal_size_scifact.json` |
| Cross-domain drift (BM25 KS up to 0.998; transferred ECE ~5–7× in-domain) | `eval/cross_dataset_drift.json` |
| E-process drift monitor | `src/conformal/sequential.py`, `eval/conformal_sequential_eprocess.json` |
| Subgroup assignment (argmax standardized input signal; label-free, no feedback from calibrated output) | `src/fusion/multicalibration.py` (`signal_dominance_subgroups`) |
| Theorem statements (CI fusion, multicalibration preservation, Mondrian–Venn–Abers validity, e-process) | `theory/proofs.tex` |

## New experiments (discussion phase)

`04_new_experiments.md` summarizes five experiments run from the frozen artifacts;
`run_rebuttal_experiments.py` reproduces them (`PYTHONPATH=. python3 rebuttal/run_rebuttal_experiments.py`,
~5 min, CPU only), writing JSON to `rebuttal/evidence/`:

| File | Contents |
|---|---|
| `e1_hallu_paired_tests.json` | Exact paired t-tests, hallu_NR full coverage, 5 subsets |
| `e2_threshold_transfer.json` | Threshold-reliability gap per stratum, 5 seeds × 7 subsets |
| `e3_selective_ndcg_ci.json` | Selective-NDCG bootstrap CIs, nfcorpus + trec-covid |
| `e4_stratum_audit.json` | Dominance-stratum sizes and fallback triggers, 7 subsets |
| `e5_scorecard.json` | Win/tie/loss vs Subgroup-Platt/-Isotonic/per-cell HKRR |

## Caution before posting

The five-subset hallucination table shows the −3.32pp effect is **scifact-specific**; on the other four subsets CalFuse-P vs Linear-Learned is within seed noise (nfcorpus/scidocs/arguana point in the other direction, n.s.). The drafts disclose this proactively — do not remove that disclosure; Reviewer utWJ's Q4 and T3MF's Q2 both probe exactly this, and the data is public in the supplementary.
