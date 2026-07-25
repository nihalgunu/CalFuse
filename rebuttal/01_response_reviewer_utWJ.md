# Response to Reviewer utWJ (rating 4, borderline accept)

Thank you for a review that asks exactly the questions a practitioner would. We answer the four questions first, then the listed weaknesses.

## Q1. "My reranker ranks fine without probabilities — why do I care?"

You don't, if your system always generates from the top-k. Calibration buys nothing for pure ranking, and we confirm this: full-coverage NDCG@10 is statistically indistinguishable between CalFuse and Linear-Learned on every subset (paired bootstrap, e.g. nfcorpus Δ=0.0005, p=0.91; arguana Δ=−0.003, p=0.61).

The paper's setting is the *thresholded* system: above τ, retrieved passages ground generation; below τ, the system abstains or falls back. A monotone score supports ranking but gives τ no semantics — τ=0.7 on an uncalibrated score is a dial you tune per corpus by trial and error, and its meaning silently changes across query subpopulations.

We ran a new experiment during the discussion period to make this concrete (all from the released frozen score matrices; 5 reseeded splits × 7 subsets). Fix τ and measure, in each dominance stratum with ≥20 pairs above τ, the **threshold-reliability gap** |realized precision − mean stated probability| — the quantity a deployment implicitly trusts when it sets "ground if p ≥ τ":

- On nfcorpus at τ=0.7, RRF's realized precision differs by up to **0.78 across strata** at the same threshold — τ literally does not mean the same thing for different query types. Its worst-stratum gap is 0.58.
- Worst-stratum gap, Linear-Learned vs CalFuse: nfcorpus 0.097 → **0.036** (paired p<0.001), scidocs (τ=0.5) 0.151 → **0.046** (p=0.010), fiqa (τ=0.7) 0.103 → **0.062** (p=0.044); significantly smaller on 4 of 7 subsets, tied on the rest, never significantly larger.

The selective curves (Fig. 4) and the hallucination experiment are the downstream cash value: on scifact, abstaining on the bottom half of CalFuse-P's calibrated scores cuts hallucination-among-answered from 30.8% to 15.2% (5-seed means), a trade a raw reranker score does not expose with known semantics. We will open Section 1 with this "when you need this / when you don't" framing and add the threshold-reliability experiment.

## Q2. "How are dominance subgroups assigned — isn't that circular?"

Assignment is a fixed, label-free function of the *input* signal vector, not of the score being calibrated. Concretely (released code, `signal_dominance_subgroups`): standardize each signal column (BM25, dense, cross-encoder families), and assign the query–passage pair to the stratum of the arg-max standardized *input* signal. Two points:

1. **No circularity.** Circularity would arise if stratum membership depended on the calibrated fused output being audited — then recalibration could move points between strata and the audit would chase its own tail. Here the partition is a measurable function of the raw inputs, is frozen before the final calibration stage, and is identical at calibration and test time; the calibrated output never feeds back into membership. This is exactly the multicalibration setting (HKRR 2018), where subgroups may be arbitrary computable functions of the covariates — including functions of features the predictor itself uses.
2. **Why this family.** Dominance strata are (a) computable at inference with no labels, (b) precisely the strata where fusion weights matter most (they select which signal's miscalibration dominates the fused logit), and (c) empirically where the failures live — that is Fig. 1. The framework accepts any user-supplied partition (intent, topic, cohort); dominance is the instantiation, not the definition.

We will add both points, with the equation for the assignment, to Section 3.

## Q3. "What does Venn–Abers add over Subgroup-Platt?"

Three things, one of which your intuition already prices in:

- **No parametric shape assumption.** Subgroup-Platt fits a 2-parameter sigmoid per stratum; when the per-stratum reliability curve isn't sigmoid, it can't fix it. We rescaled this comparison to 20 reseeded splits during the discussion period (4× the paper's protocol, from the frozen matrices): per-stratum IVAP beats Subgroup-Platt on worst-subgroup ECE-15 on 4 of 7 subsets (fiqa and arguana p<0.0001, scifact p=0.0004, nfcorpus p=0.0008), ties the other 3, loses none. Subgroup-Platt is a good baseline that fails exactly when the miscalibration is shape-wise nontrivial.
- **Finite-sample validity.** Per-stratum IVAP carries a distribution-free calibration guarantee under exchangeability; Platt carries none.
- **The envelope.** IVAP returns `[p_lo, p_hi]`, giving a per-prediction uncertainty width usable for abstention and monitoring; Platt gives a point.

Full transparency, matching Reviewer T3MF's observation: against Subgroup-*Isotonic* the point estimate is close — at 20 seeds IVAP wins on 2 of 7 subsets (trec-covid p=0.005, touche-2020 p<0.0001), ties the other 5, loses none. The revision will state plainly that the bulk of the point-estimate gain comes from stratified *monotone nonparametric* calibration, and that IVAP's main additional contribution is the validity guarantee and the envelope.

## Q4. "Is the 3.32pp hallucination result robust?"

Your skepticism is warranted, and we will restructure this section. The released artifacts already contain the same 5-seed protocol on **five** subsets (nfcorpus, scifact, fiqa, arguana, scidocs), not just scifact. Hallucination-among-non-refused at full coverage, 5-seed mean ± sd, CalFuse-P vs Linear-Learned (strongest learned baseline):

| subset | CalFuse-P | Linear-Learned | paired p |
|---|---|---|---|
| scifact | **30.8 ± 3.7** | 34.1 ± 4.4 | **0.045** (d=−1.29) |
| fiqa | 61.2 ± 3.0 | 62.4 ± 8.6 | 0.75 |
| nfcorpus | 72.3 ± 5.6 | 70.1 ± 7.2 | 0.29 |
| scidocs | 50.0 ± 7.5 | 47.7 ± 10.2 | 0.44 |
| arguana | 75.6 ± 5.2 | 72.4 ± 7.1 | 0.23 |

Only scifact is significant; the other four are within seed noise in both directions (per-method values for every seed are in the released `eval/multiseed/`). The honest claim, which the revision will make, is therefore mechanistic, not universal: *when* the top-k composition is contested (scifact: short claims, dense gold evidence), better subgroup calibration swaps confident distractors out of the context and hallucination drops measurably; where composition is saturated or the generator's failure mode is not retrieval-driven, calibration cannot help and doesn't. We will present all five subsets in the main text, keep scifact as the mechanism case study (rank-of-first-positive distributions are unchanged across methods — the improvement is provably not ranking sharpening), and soften the abstract accordingly. Scaling beyond Qwen-2.5-7B is future work; we will say so rather than imply generality.

## Weaknesses 3 & 4 (rare strata; exchangeability)

- **Rare strata:** the released implementation guards this: a stratum below a size floor (30 pairs) or a positive-count floor falls back to the *pooled* IVAP, so the method degrades to marginal Venn–Abers, never below it. A new audit across all seven subsets shows calibration-split strata range from 245 to 6,686 pairs and the fallback never fires in our benchmarks — the safeguard is for deployments with genuinely rare strata, not load-bearing in the reported results. The calibration-size ablation supports graceful degradation: on scifact with only 10% of calibration data (15 queries / 616 pairs), worst-subgroup ECE-15 is 0.023 vs 0.020–0.022 at full size.
- **Exchangeability:** agreed — the guarantee is per-domain. We quantify how badly it breaks across domains (cross-dataset transfer: per-signal KS drift up to 0.998 for BM25, and transferred ECE inflates ~5–7×), and the paper ships an anytime-valid e-process monitor that detects calibration drift in deployment, upon which recalibration needs only a fresh labeled calibration split (no retriever retraining). We will promote this from appendix to the Limitations discussion.
