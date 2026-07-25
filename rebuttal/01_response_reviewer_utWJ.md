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

## Q4. "Is the 3.32pp hallucination result robust? Does it transfer to bigger models?"

Your skepticism was warranted, and during the discussion period we ran the two experiments the question demands: (a) **doubled every seed** (5→10) on the published Qwen-2.5-7B protocol across all five subsets with verdicts, and (b) reran the three key subsets on **Qwen-2.5-14B-Instruct** (5 seeds). Hallucination-among-non-refused at full coverage, CalFuse-P vs Linear-Learned:

**Qwen-7B, 10 seeds:**

| subset | CalFuse-P | Linear-Learned | paired p |
|---|---|---|---|
| scifact | **30.6 ± 4.3** | 34.1 ± 4.0 | **0.0025** (d=−1.31) |
| fiqa | 58.6 ± 5.0 | 61.4 ± 8.0 | 0.19 |
| nfcorpus | 71.3 ± 5.1 | 68.5 ± 5.9 | **0.014** (CalFuse-P *worse*) |
| scidocs | 54.4 ± 7.8 | 51.7 ± 8.2 | 0.13 |
| arguana | 77.2 ± 6.1 | 76.0 ± 7.8 | 0.42 |

The scifact effect is real and strengthens with power: −3.49 pp at p=0.0025, no longer borderline. Equally honestly: at 10 seeds a significant *deficit* emerges for CalFuse-P on nfcorpus (+2.8 pp, p=0.014); the full CalFuse (with the stratified nonparametric stage) shows no deficit there (68.4 vs 68.5, p=0.96). So the per-subset picture is one robust win, one loss for the parametric base that the full method repairs to parity, and three inconclusive.

**Qwen-14B, 5 seeds (scifact, fiqa, nfcorpus):** every between-method difference collapses (all |Δ| ≤ 0.6 pp, all p>0.69; e.g. scifact CalFuse-P 23.2 vs Linear-Learned 22.8), while overall hallucination drops (scifact ~34%→~23%). The direct answer to "does it transfer to bigger models": **no** — at 14B the generator is robust enough to distractor composition that fusion-method differences vanish end-to-end.

The revision will report all of this and scope the claim accordingly: better subgroup calibration changes *which* distractors enter the context (the mechanism test stands — rank-of-first-positive is unchanged), and this measurably reduces hallucination for a 7B-class generator on contested-composition subsets, with the effect washing out both for stronger generators and for subsets where composition isn't the failure mode. The paper's primary contribution — worst-subgroup calibration, where the 20-seed rescaling strengthens every claim — does not rest on the hallucination experiment; the experiment now demarcates where calibration does and does not reach end-to-end behavior, which we believe is more useful to practitioners than an unscoped claim.

## Weaknesses 3 & 4 (rare strata; exchangeability)

- **Rare strata:** the released implementation guards this: a stratum below a size floor (30 pairs) or a positive-count floor falls back to the *pooled* IVAP, so the method degrades to marginal Venn–Abers, never below it. A new audit across all seven subsets shows calibration-split strata range from 245 to 6,686 pairs and the fallback never fires in our benchmarks — the safeguard is for deployments with genuinely rare strata, not load-bearing in the reported results. The calibration-size ablation supports graceful degradation: on scifact with only 10% of calibration data (15 queries / 616 pairs), worst-subgroup ECE-15 is 0.023 vs 0.020–0.022 at full size.
- **Exchangeability:** agreed — the guarantee is per-domain. We quantify how badly it breaks across domains (cross-dataset transfer: per-signal KS drift up to 0.998 for BM25, and transferred ECE inflates ~5–7×), and the paper ships an anytime-valid e-process monitor that detects calibration drift in deployment, upon which recalibration needs only a fresh labeled calibration split (no retriever retraining). We will promote this from appendix to the Limitations discussion.
