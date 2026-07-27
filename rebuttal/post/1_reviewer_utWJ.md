Thank you for asking exactly the questions a practitioner would.

**Q1. "My reranker ranks fine without probabilities — why do I care?"**

You don't, if your system always generates from the top-k: full-coverage NDCG@10 is statistically indistinguishable between CalFuse and Linear-Learned on every subset (e.g. nfcorpus Δ=0.0005, p=0.91). The paper's setting is the *thresholded* system — above τ, retrieved passages ground generation; below τ, the system abstains — where a monotone score gives τ no semantics, and its meaning silently changes across query subpopulations.

New discussion-period experiment (20 reseeded splits × 7 subsets, from the frozen score matrices): fix τ; in each dominance stratum with ≥20 pairs above τ, measure the **threshold-reliability gap** |realized precision − mean stated probability| — the quantity a deployment trusts when it sets "ground if p ≥ τ":

- On nfcorpus at τ=0.7, RRF's realized precision differs by up to **0.80 across strata** at the same threshold; its worst-stratum gap is 0.61.
- Worst-stratum gap, Linear-Learned vs CalFuse: nfcorpus (τ=0.7) 0.075 → **0.036** (paired p<0.0001, d=−1.2); scidocs (τ=0.5) 0.170 → **0.067** (p<0.0001, d=−1.6); fiqa trend (p=0.095); tied elsewhere, never significantly larger.

On scifact, abstaining on the bottom half of calibrated scores cuts hallucination-among-answered from 30.8% to 15.2%. Section 1 will open with this framing, plus the new experiment.

**Q2. Subgroup assignment — circular?**

No: assignment is a fixed, label-free function of the *input* signal vector — standardize each signal column, assign the pair to the stratum of the arg-max standardized input signal. Circularity would require membership to depend on the calibrated fused output; here the partition is frozen before the final calibration stage, identical at calibration/test time, and the output never feeds back. This is the standard multicalibration setting (HKRR 2018): subgroups may be arbitrary computable functions of the covariates, including features the predictor uses. We chose dominance strata because they are (a) label-free at inference, (b) where fusion weights matter most, (c) empirically where the failures live (Fig. 1). The framework accepts any partition; dominance is the instantiation. Both points and the assignment equation go into Section 3.

**Q3. What does Venn–Abers add over Subgroup-Platt?**

- *No shape assumption.* Subgroup-Platt fits a 2-parameter sigmoid per stratum. At 20 reseeded splits, per-stratum IVAP beats it on worst-subgroup ECE-15 on 4 of 7 subsets (fiqa, arguana p<0.0001; scifact p=0.0004; nfcorpus p=0.0008), ties 3, loses none — it fails exactly when miscalibration is shape-wise nontrivial.
- *Finite-sample validity* under exchangeability; Platt has none.
- *The envelope* [p_lo, p_hi] for abstention and monitoring; Platt gives a point.

Transparency, matching Reviewer T3MF: against Subgroup-*Isotonic*, IVAP wins on 2 of 7 (trec-covid p=0.005, touche p<0.0001), ties 5, loses none. The bulk of the point-estimate gain comes from stratified monotone calibration; IVAP's addition is the guarantee and envelope.

**Q4. Is −3.32pp robust? Does it transfer to bigger models?**

We ran both experiments the question demands: (a) **10 seeds** (doubled) on the published Qwen-7B protocol, all five subsets; (b) three subsets on **Qwen-2.5-14B**. Hallucination-among-non-refused, CalFuse-P vs Linear-Learned (7B, 10 seeds):

| subset | CalFuse-P | Linear-L | p |
|---|---|---|---|
| scifact | **30.6±4.3** | 34.1±4.0 | **0.0025** (d=−1.31) |
| fiqa | 58.6±5.0 | 61.4±8.0 | 0.19 |
| nfcorpus | 71.3±5.1 | 68.5±5.9 | **0.014** (*worse*) |
| scidocs | 54.4±7.8 | 51.7±8.2 | 0.13 |
| arguana | 77.2±6.1 | 76.0±7.8 | 0.42 |

scifact strengthens with power — no longer borderline. Equally honestly: a significant deficit emerges for CalFuse-P on nfcorpus, which the full CalFuse repairs to parity (68.4 vs 68.5, p=0.96). At **14B**, every between-method difference collapses (|Δ|≤0.6pp, p>0.69) while overall hallucination drops (~34%→~23% on scifact). Direct answer: **no, it does not transfer** — the effect is a 7B-scale phenomenon on contested-composition subsets. The revision reports and scopes all of this; the primary contribution — worst-subgroup calibration, strengthened at 20 seeds — does not rest on it.

**Weaknesses 3–4.** *Rare strata:* strata below 30 pairs (or a positive floor) fall back to pooled IVAP — never worse than marginal; a new audit shows strata of 245–6,686 pairs across all subsets (fallback never fires); with 10% calibration data, worst-subgroup ECE is 0.023 vs ~0.021 full. *Exchangeability:* agreed — per-domain guarantee; we quantify cross-domain breakage (KS up to 0.998; transferred ECE ~5–7×) and ship an anytime-valid e-process drift monitor; recalibration needs only a fresh labeled calibration split. Promoted to Limitations.

May we ask whether these answers resolve your concerns, particularly Q4?
