# Response to Reviewer T3MF (rating 4, borderline accept)

Thank you for the most complete reading of the paper we received — your summary of what the ablation does and does not establish is exactly right, and we adopt your framing in the revision.

## Q1. How should the lower bound be interpreted relative to CalFuse?

Your reading is correct and we will make it the paper's official one. Precisely:

- Theorem 1 lower-bounds the worst-cell sample complexity at Θ(α⁻² log K) for the class F_n of post-hoc calibrators whose per-cell output depends on calibration data only through per-cell empirical means. This matches HKRR's upper bound, so within F_n the analysis is tight: the per-cell route is *optimally* sample-hungry, and no cleverness inside that class escapes the cost.
- The theorem therefore functions as a design justification — the reason CalFuse pools strength across cells via per-stratum isotonic structure instead of auditing cells independently — and not as a performance guarantee for CalFuse. CalFuse's direct guarantees are of a different type: distribution-free per-stratum coverage of the Venn–Abers envelope under exchangeability, and multicalibration preservation of the parametric fusion stage under conditional independence.
- New supporting evidence computed during the discussion period: our HKRR-style per-cell corrector baseline (`calfuse_multical`) is precisely an *empirical instantiation of the class F_n* the theorem characterizes — and on worst-subgroup ECE-15 it loses to CalFuse on 3 of 7 subsets and never wins, a result we verified at 20 reseeded splits (nfcorpus p<0.0001, fiqa p=0.017, touche-2020 p=0.002). The lower bound's prediction (per-cell audits are worst-cell sample-starved at these calibration-set sizes) is visible in the data, which tightens the theory–method link from "motivation" to "motivation + confirmed empirical consequence".
- What is missing, and will now be stated as an open problem in the Limitations: a matching *upper* bound on worst-subgroup calibration error for stratified monotone calibration, which would close the loop by showing the escape from F_n buys the predicted improvement in rates rather than merely avoiding the lower bound's hypothesis. We will retitle the theory section "Why per-cell post-hoc calibration cannot suffice" so no reader mistakes the bound for a guarantee about CalFuse.

## Q2. What can and cannot be concluded from the hallucination experiment?

We will substantially sharpen this. First, transparency: the released artifacts contain the identical 5-seed protocol on five subsets (nfcorpus, scifact, fiqa, arguana, scidocs). At full coverage, CalFuse-P vs the strongest learned baseline (Linear-Learned) on hallucination-among-non-refused: scifact −3.3pp (p=0.045, d=−1.29); the other four subsets are within seed noise in both directions (e.g. fiqa −1.2pp, nfcorpus +2.2pp, all n.s.). The revision will show all five.

What can be concluded: on a subset where top-k composition is contested, improving subgroup calibration changes *which* passages enter the context, and the direct mechanism test (rank-of-first-positive distributions statistically unchanged across methods; top-k composition changed) rules out ranking sharpening — so the reduction is attributable to distractor swap, which is the causal pathway a calibration method could honestly claim. What cannot be concluded: a general "calibration reduces hallucination" effect across corpora, models, or scales. We view the experiment as an existence-plus-mechanism demonstration and will label it as such, with model scale (Qwen-2.5-7B only) listed as an explicit external-validity limit.

## Q3. Selective-NDCG on denser-relevance data

Agreed that trec-covid with n_q=15 test queries carries the signature but not much evidential weight. We ran both suggested checks during the discussion period (from the frozen score matrices):

- **Bootstrap CIs on trec-covid** (2,000 query resamples): selective-NDCG AUC difference CalFuse − Linear-Learned = +0.027, 95% CI [−0.031, +0.109], with 70% of bootstrap mass positive. Suggestive, not significant — exactly as your reading implied.
- **Secondary denser-judgment subset (nfcorpus)**: −0.008, CI [−0.020, +0.004] — the signature is absent.

Accordingly the revision downgrades this thread explicitly: the signature is stated as a prediction of the tower-property derivation (given *before* the result, so it reads as a test rather than post-hoc selection), reported as suggestive on trec-covid and absent on nfcorpus with CIs shown, and TREC DL 2019–2021 passage tracks (dense judgments, hundreds of queries) named as the decisive test — camera-ready if accepted, designated follow-up otherwise. We believe reporting the null secondary check verbatim is the right precedent for this kind of claim.

## On the main weakness: the operative piece is stratified monotone calibration, not the envelope

We accept this characterization, with one refinement from a 20-seed rescaling run during the discussion period: against Subgroup-Isotonic the IVAP point estimate wins on 2 of 7 subsets (trec-covid p=0.005, touche-2020 p<0.0001), ties the other five, and never loses — so "mostly tied" remains the honest summary, and the envelope-based abstention experiments show no reliable advantage over point-estimate ranking (envelope-lower-bound ordering helps only at very low coverage on some subsets, within noise). The revision will (i) state in the contributions that the point-estimate gains are attributable to *stratified nonparametric monotone* calibration, (ii) reposition IVAP's contribution as the finite-sample validity guarantee and the per-prediction width — properties isotonic alone lacks and which cost nothing extra at inference (both calibrators are a sorted-array lookup) — and (iii) move the envelope-abstention comparison into the main ablation table so the tie is visible rather than implied. We believe the honest version of the claim is still a contribution: the guarantee is free given the mechanism that already wins on ECE.

We will also follow your suggestion to demote the secondary evaluation threads (selective-NDCG, envelope abstention) to clearly-marked supporting evidence so the central calibration story is not diluted.
