# Response to Reviewer utWJ (rating 4, borderline accept)

Thank you for a review that asks exactly the questions a practitioner would. We answer the four questions first, then the listed weaknesses.

## Q1. "My reranker ranks fine without probabilities — why do I care?"

You don't, if your system always generates from the top-k. Calibration buys nothing for pure ranking, and we confirm this: full-coverage NDCG@10 is statistically indistinguishable between CalFuse and Linear-Learned on every subset (paired bootstrap, e.g. nfcorpus Δ=0.0005, p=0.91; arguana Δ=−0.003, p=0.61).

The paper's setting is the *thresholded* system: above τ, retrieved passages ground generation; below τ, the system abstains or falls back. A monotone score supports ranking but gives τ no semantics — τ=0.7 on an uncalibrated score is a dial you tune per corpus by trial and error, and its meaning silently changes across query subpopulations. A calibrated score makes τ a statement about error rates ("ground only when P(relevant) ≥ 0.7"), which is what lets one policy transfer across subgroups — and subgroup calibration is precisely what fails for the standard pipelines (marginal ECE understates worst-subgroup ECE by 1.4–7.7× for fusion baselines; e.g. RRF on fiqa: 0.011 marginal vs 0.086 worst-subgroup). The selective curves (Fig. 4) and the hallucination experiment are the downstream cash value: on scifact, abstaining on the bottom half of CalFuse-P's calibrated scores cuts hallucination-among-answered from 30.8% to 15.2% (5-seed means), a trade a raw reranker score does not expose with known semantics. We will open Section 1 with this "when you need this / when you don't" framing.

## Q2. "How are dominance subgroups assigned — isn't that circular?"

Assignment is a fixed, label-free function of the *input* signal vector, not of the score being calibrated. Concretely (released code, `signal_dominance_subgroups`): standardize each signal column (BM25, dense, cross-encoder families), and assign the query–passage pair to the stratum of the arg-max standardized *input* signal. Two points:

1. **No circularity.** Circularity would arise if stratum membership depended on the calibrated fused output being audited — then recalibration could move points between strata and the audit would chase its own tail. Here the partition is a measurable function of the raw inputs, is frozen before the final calibration stage, and is identical at calibration and test time; the calibrated output never feeds back into membership. This is exactly the multicalibration setting (HKRR 2018), where subgroups may be arbitrary computable functions of the covariates — including functions of features the predictor itself uses.
2. **Why this family.** Dominance strata are (a) computable at inference with no labels, (b) precisely the strata where fusion weights matter most (they select which signal's miscalibration dominates the fused logit), and (c) empirically where the failures live — that is Fig. 1. The framework accepts any user-supplied partition (intent, topic, cohort); dominance is the instantiation, not the definition.

We will add both points, with the equation for the assignment, to Section 3.

## Q3. "What does Venn–Abers add over Subgroup-Platt?"

Three things, one of which your intuition already prices in:

- **No parametric shape assumption.** Subgroup-Platt fits a 2-parameter sigmoid per stratum; when the per-stratum reliability curve isn't sigmoid, it can't fix it. This is visible in the data: on fiqa, Subgroup-Platt vs per-stratum IVAP on worst-subgroup ECE-15 differs by +0.019 (paired t over 5 seeds, p<0.001, d=3.9); scifact p=0.043, d=1.3; on subsets where the curve happens to be near-sigmoid (nfcorpus) the two tie (p=0.21). So Subgroup-Platt is a good baseline that fails exactly when the miscalibration is shape-wise nontrivial.
- **Finite-sample validity.** Per-stratum IVAP carries a distribution-free calibration guarantee under exchangeability; Platt carries none.
- **The envelope.** IVAP returns `[p_lo, p_hi]`, giving a per-prediction uncertainty width usable for abstention and monitoring; Platt gives a point.

Full transparency, matching Reviewer T3MF's observation: against Subgroup-*Isotonic* the point estimate is statistically tied on all seven subsets. The revision will state plainly that the point-estimate gain comes from stratified *monotone nonparametric* calibration, and that IVAP's marginal contribution over isotonic is the guarantee and the envelope, not additional ECE.

## Q4. "Is the 3.32pp hallucination result robust?"

Your skepticism is warranted, and we will restructure this section. The released artifacts already contain the same 5-seed protocol on **five** subsets (nfcorpus, scifact, fiqa, arguana, scidocs), not just scifact. Hallucination-among-non-refused at full coverage, 5-seed mean ± sd, CalFuse-P vs Linear-Learned (strongest learned baseline):

| subset | CalFuse-P | Linear-Learned |
|---|---|---|
| scifact | **30.8 ± 3.7** | 34.1 ± 4.4 |
| fiqa | 61.2 ± 3.0 | 62.4 ± 8.6 |
| nfcorpus | 72.3 ± 5.6 | 70.1 ± 7.2 |
| scidocs | 50.0 ± 7.5 | 47.7 ± 10.2 |
| arguana | 75.6 ± 5.2 | 72.4 ± 7.1 |

Only scifact is significant (p=0.045, d=−1.29); the other four are within seed noise in both directions. The honest claim, which the revision will make, is therefore mechanistic, not universal: *when* the top-k composition is contested (scifact: short claims, dense gold evidence), better subgroup calibration swaps confident distractors out of the context and hallucination drops measurably; where composition is saturated or the generator's failure mode is not retrieval-driven, calibration cannot help and doesn't. We will present all five subsets in the main text, keep scifact as the mechanism case study (rank-of-first-positive distributions are unchanged across methods — the improvement is provably not ranking sharpening), and soften the abstract accordingly. Scaling beyond Qwen-2.5-7B is future work; we will say so rather than imply generality.

## Weaknesses 3 & 4 (rare strata; exchangeability)

- **Rare strata:** the released implementation guards this: a stratum below a size floor (30 pairs) or a positive-count floor falls back to the *pooled* IVAP, so the method degrades to marginal Venn–Abers, never below it. The calibration-size ablation supports graceful degradation: on scifact with only 10% of calibration data (15 queries / 616 pairs), worst-subgroup ECE-15 is 0.023 vs 0.020–0.022 at full size.
- **Exchangeability:** agreed — the guarantee is per-domain. We quantify how badly it breaks across domains (cross-dataset transfer: per-signal KS drift up to 0.998 for BM25, and transferred ECE inflates ~5–7×), and the paper ships an anytime-valid e-process monitor that detects calibration drift in deployment, upon which recalibration needs only a fresh labeled calibration split (no retriever retraining). We will promote this from appendix to the Limitations discussion.
