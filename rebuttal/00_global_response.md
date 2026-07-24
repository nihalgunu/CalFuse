# Global response (AC Sraq + all reviewers)

We thank the AC and all three reviewers for careful, constructive reviews. Below we summarize the two cross-cutting concerns and the concrete revisions we commit to; per-reviewer replies contain the details and new numbers.

## 1. Self-containedness and jargon (AC, Reviewer utWJ)

We agree the submission leaned too hard on unexpanded shorthand. The revision will make the paper self-contained:

- **Platt scaling** will be defined at first use (a two-parameter logistic map `p = σ(a·s + b)` fit on held-out calibration data; Platt, 1999), with the reference added.
- **IVAP (Inductive Venn–Abers Predictor)** will be defined in one paragraph before Section 4: a nonparametric calibrator that fits isotonic regression twice on the calibration set — once appending the test point with label 0, once with label 1 — yielding an interval `[p_lo, p_hi]` whose validity holds distribution-free under exchangeability (Vovk & Petej, 2014). The point estimate is a fixed merge of the two.
- **Baseline glossary.** A table will define every baseline at first mention: single-retriever + Platt (BM25, BGE, E5, cross-encoder, each Platt-scaled on the calibration split), RRF (reciprocal rank fusion, Cormack et al., 2009, followed by Platt), Linear-Learned (logistic regression over per-signal calibrated logits), Subgroup-Platt (Linear-Learned followed by a separate Platt map per dominance stratum), Subgroup-Isotonic (same with isotonic regression per stratum).
- **"Distractor swap"** will be defined where it first appears: at fixed rank-of-first-positive, better-calibrated fusion changes *which non-relevant passages* fill the remaining top-k slots, replacing confident topical distractors (which the LLM paraphrases as if grounded) with passages the generator is likelier to ignore or refuse on. We will present the mechanism test (rank distributions unchanged; top-k composition changed) alongside the definition rather than after it.
- A notation table (τ, ECE-15, worst-subgroup ECE, stratum g, K cells, α) will be added to Section 2.

## 2. Role of the theory (all three reviewers)

All three reviews raise, in different forms, how Theorem 1 relates to CalFuse. We will add an explicit paragraph, "What Theorem 1 does and does not say":

- **It does say:** any post-hoc calibrator in the class F_n — methods whose per-cell output depends on calibration data only through per-cell empirical means (per-cell recalibration, HKRR-style per-cell audits, binned Subgroup-Platt) — must pay Θ(α⁻² log K) calibration samples *in the worst cell*. This matches HKRR's upper bound, so within F_n the per-cell route is optimally sample-hungry: the cost is real, not an artifact of loose analysis.
- **It does not say:** anything about CalFuse's own sample complexity or optimality. CalFuse is deliberately outside F_n: within each stratum, isotonic pooling shares strength across cells instead of estimating each cell mean independently. The theorem is the *reason to leave the class*, not a guarantee about the method that leaves it.
- **What CalFuse does carry directly:** finite-sample, distribution-free per-stratum validity of the Venn–Abers envelope under exchangeability (Mondrian–Venn–Abers validity theorem), multicalibration preservation of the parametric fusion stage under conditional independence, and an anytime-valid e-process for monitoring the exchangeability assumption in deployment. We will state explicitly that CalFuse's guarantees are validity/coverage guarantees, and that a matching upper bound for stratified monotone calibration is open (we now say so in Limitations).

## 3. New material added during the discussion phase

- The full **five-subset × five-seed LLM hallucination table** (nfcorpus, scifact, fiqa, arguana, scidocs; all runs were already in the released artifacts) with honest discussion of where the effect does and does not appear — see reply to Reviewer utWJ (Q4) and Reviewer T3MF (Q2).
- The Subgroup-Platt / Subgroup-Isotonic / IVAP three-way comparison with paired statistics — see reply to Reviewer utWJ (Q3).
- Clarified, non-circular definition of dominance strata — see reply to Reviewer utWJ (Q2).

All scores, verdicts, and analysis scripts behind these numbers are in the released supplementary code, so every number below is reproducible from the frozen artifacts without re-running retrievers or the LLM.
