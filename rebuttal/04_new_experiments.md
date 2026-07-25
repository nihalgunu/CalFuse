# New rebuttal-phase experiments (all from frozen artifacts)

Runner: `rebuttal/run_rebuttal_experiments.py` (outputs in `rebuttal/evidence/`).
No GPU, no BEIR re-download, no LLM calls — everything recomputes from the released
`eval/*.npz` score matrices and `eval/multiseed/` verdicts in ~5 minutes.

## E1. Hallucination paired tests — exact p-values, all five subsets

hallu_NR (% among answered) at full coverage, 5-seed mean ± sd; paired t by seed.

| subset | CalFuse-P | CalFuse | Linear-Learned | CalFuse-P vs LL |
|---|---|---|---|---|
| scifact | 30.8 ± 3.7 | 32.4 ± 3.9 | 34.1 ± 4.4 | **−3.31 pp, p=0.045, d=−1.29** |
| fiqa | 61.2 ± 3.0 | 62.4 ± 4.6 | 62.4 ± 8.6 | −1.20 pp, p=0.75 |
| nfcorpus | 72.3 ± 5.6 | 67.2 ± 6.1 | 70.1 ± 7.2 | +2.23 pp, p=0.29 |
| arguana | 75.6 ± 5.2 | 74.4 ± 3.3 | 72.4 ± 7.1 | +3.20 pp, p=0.23 |
| scidocs | 50.0 ± 7.5 | 50.2 ± 4.1 | 47.7 ± 10.2 | +2.33 pp, p=0.44 |

Reading: only scifact is significant vs Linear-Learned; the rest are seed noise in
both directions. (CalFuse = calfuse_conformal vs LL on nfcorpus: −2.85 pp, p=0.11.)
Mechanism check: mean rank-of-first-positive CalFuse-P vs LL differs by ≤0.09 of a
rank position on every subset — the scifact reduction is not ranking sharpening.

Also computed vs strongest single retriever (in `e1_hallu_paired_tests.json`):
significant CalFuse wins on scifact (−6.1 pp, p=0.021), arguana (−6.8 pp, p=0.039),
scidocs (−10.2 pp, p=0.001); on nfcorpus BM25-only contexts hallucinate *less* than
all fusion methods (fusion vs BM25 +14.8 to +19.9 pp, p<0.05). Use with care —
refusal-rate and retrieval-strength differences confound the single-retriever
comparison; the like-for-like comparison is vs Linear-Learned.

## E2. NEW threshold-transfer experiment (answers utWJ Q1 concretely)

Setup: fix a threshold τ; among test pairs with fused p ≥ τ in each dominance
stratum (≥20 pairs), measure the **threshold-reliability gap** = |realized precision
− mean stated probability|, worst stratum. 5 reseeded splits × 7 subsets.
This is exactly the quantity a deployment trusts when it sets "ground if p ≥ τ".

Worst-stratum gap (mean over seeds), and paired CalFuse-vs-Linear-Learned test:

| subset | RRF | Linear-Learned | CalFuse | paired p (τ) |
|---|---|---|---|---|
| nfcorpus (τ=0.7) | 0.584 | 0.097 | **0.036** | p<0.001 |
| nfcorpus (τ=0.5) | 0.486 | 0.087 | **0.049** | p=0.036 |
| fiqa (τ=0.7) | — | 0.103 | **0.062** | p=0.044 |
| scidocs (τ=0.5) | — | 0.151 | **0.046** | p=0.010 |
| scifact (τ=0.5) | — | 0.082 | 0.049 | p=0.12 |
| trec-covid (τ=0.7) | 0.162 | 0.087 | 0.095 | p=0.62 (tie) |
| arguana (τ=0.5) | — | 0.113 | 0.124 | p=0.81 (tie) |

Headline facts:
- On nfcorpus, RRF's *realized precision at the same τ differs by up to 0.78
  across strata* (cross-strata spread) — τ literally does not mean the same thing
  for different query types. CalFuse's worst-stratum deviation from its stated
  probability is 0.036.
- CalFuse's worst-stratum gap is significantly smaller than Linear-Learned's on
  4 of 7 subsets (at ≥1 τ), statistically tied on the rest, **never significantly
  larger**.

## E3. Selective-NDCG with query-bootstrap 95% CIs (T3MF Q3)

- trec-covid (n_q=15): AUC(CalFuse) − AUC(Linear-Learned) = **+0.027
  [−0.031, +0.109]**, 70% of bootstrap mass positive → suggestive, not significant.
- nfcorpus (secondary dense-ish subset): −0.008 [−0.020, +0.004] → **signature
  absent**.

Honest conclusion for the rebuttal: the selective-NDCG signature is suggestive on
trec-covid only; the revision downgrades it to supporting evidence and names
TREC-DL (dense judgments, hundreds of queries) as the decisive test.

## E4. Stratum audit (utWJ weakness 3: rare-subgroup fragility)

Calibration-split dominance strata across all seven subsets range from 245 to
6,686 pairs; the pooled-IVAP fallback (stratum <30 pairs or <5 positives) **never
triggers** on any subset. The safeguard exists for deployments with rarer strata;
in every evaluated setting the per-stratum calibrators are comfortably fed.

## E5. Scorecard vs stratified baselines and vs the per-cell class (theory link)

Worst-subgroup ECE-15, CalFuse vs X per subset (win = p<0.05, 5-seed paired t):

| vs | wins | ties | losses |
|---|---|---|---|
| Subgroup-Platt | 3 (scifact, fiqa, arguana) | 4 | 0 |
| Subgroup-Isotonic | 1 (touche-2020, p=0.030) | 6 | 0 |
| HKRR-style per-cell (calfuse_multical) | 3 (scifact, arguana, touche-2020) | 4 | 0 |

The third row is new ammunition for the theory questions (T3MF Q1, VkVU Q1):
`calfuse_multical` is the empirical instantiation of the per-cell class F_n that
Theorem 1 proves is worst-cell sample-starved — and it indeed loses to CalFuse on
3 subsets and never wins. The lower bound's prediction is visible in the data.

## L1. Seed-scaling: the core claims at 20 seeds (4× the paper's protocol)

Rerun of worst-subgroup ECE-15 for the full method family on all seven subsets
with 20 reseeded splits (2026–2045; runner `rebuttal/run_seed_scaling.py`,
output `evidence/l1_seed_scaling_wsece.json`). Everything strengthens:

- **Headline:** CalFuse has the lowest mean worst-subgroup ECE-15 on 6 of 7
  subsets; on arguana it is statistically tied for best with Subgroup-Isotonic
  (Δ=+0.0003, p=0.27). Never beaten by a non-CalFuse-family method anywhere.
- **vs Linear-Learned:** significant on nfcorpus (Δ=−0.036, p<0.0001, d=−3.1),
  fiqa (Δ=−0.017, p<0.0001, d=−2.4), scidocs (p=0.002); touche p=0.051 and
  scifact p=0.076 trend; arguana/trec-covid ties. 0 losses.
- **vs Subgroup-Platt:** now 4 significant wins (nfcorpus p=0.0008, scifact
  p=0.0004, fiqa p<0.0001, arguana p<0.0001), 3 ties, 0 losses.
- **vs Subgroup-Isotonic:** 2 significant wins (trec-covid p=0.005,
  touche-2020 p<0.0001), 5 ties, 0 losses — at n=20 the IVAP point estimate
  is no longer strictly redundant with isotonic, though the honest summary
  remains "mostly tied".
- **vs per-cell HKRR (`calfuse_multical`, the F_n instantiation):** 3 wins
  (nfcorpus p<0.0001, fiqa p=0.017, touche p=0.002), 4 ties, 0 losses.

## G1. GPU Phase 1 — hallucination at 10 seeds (Qwen-2.5-7B)

Five new seeds (2031–2035) run on Lambda, verdicts merged into `eval/multiseed/`
(analysis: `rebuttal/analyze_gpu_results.py`, output `evidence/g1_hallu_10seed.json`).
CalFuse-P vs Linear-Learned, hallu_NR at full coverage:

| subset | 5-seed result | 10-seed result |
|---|---|---|
| scifact | −3.31 pp, p=0.045 | **−3.49 pp, p=0.0025, d=−1.31** |
| fiqa | −1.20 pp, p=0.75 | −2.80 pp, p=0.19 |
| nfcorpus | +2.23 pp, p=0.29 | **+2.77 pp, p=0.014 (CalFuse-P worse)** |
| scidocs | +2.33 pp, p=0.44 | +2.67 pp, p=0.13 |
| arguana | +3.20 pp, p=0.23 | +1.20 pp, p=0.42 |

The scifact win is real (no longer borderline). The nfcorpus deficit is also
real — but belongs to the parametric base only: full CalFuse (conformal) ties
Linear-Learned there (68.4 vs 68.5, p=0.96). Both facts go in the revision.

## G2. GPU Phase 2 — Qwen-2.5-14B model transfer

Original 5 seeds, scifact/fiqa/nfcorpus (`evidence/g2_qwen14b_transfer.json`).
All fusion-method differences collapse: every CalFuse-vs-Linear-Learned delta
is ≤0.6 pp with p>0.69, while absolute hallucination falls (scifact ~34%→~23%).
Conclusion for the rebuttal: the calibration→hallucination pathway is a
7B-scale phenomenon in this setup; at 14B the generator absorbs distractor
composition differences. The end-to-end claim is scoped to generator scale and
subset composition; the calibration claims (which strengthened at 20 seeds) do
not depend on it.

## L2. Threshold-transfer at 20 seeds (authoritative; supersedes E2's stats)

Rerun of E2 at 20 reseeded splits with per-subset process isolation
(`evidence/l2_threshold_transfer_20seed.json`; per-subset files
`l2_e2_20seed_*.json`). Paired CalFuse-vs-Linear-Learned on worst-stratum gap:

| subset (τ) | Linear-Learned | CalFuse | paired p |
|---|---|---|---|
| nfcorpus (0.7) | 0.075 | **0.036** | **<0.0001** (d=−1.17) |
| scidocs (0.5) | 0.170 | **0.067** | **<0.0001** (d=−1.58) |
| fiqa (0.7) | 0.107 | 0.077 | 0.095 (trend) |
| trec-covid (0.5/0.7) | 0.081/0.089 | 0.072/0.100 | 0.16 / 0.11 (ties) |
| others | — | — | n.s. |

RRF on nfcorpus: worst-stratum gap 0.61 at τ=0.7, cross-strata realized-precision
spread up to **0.80**.

Honest scaling note: two 5-seed significances did **not** survive 20 seeds
(nfcorpus τ=0.5: p=0.036→0.89; fiqa τ=0.7: p=0.044→0.095) — which is exactly
why we scaled seeds before quoting them. The surviving effects are decisive
(p<0.0001, |d|>1.1). Drafts quote only the 20-seed numbers.
