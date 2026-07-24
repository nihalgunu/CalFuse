# Response to Reviewer T3MF (rating 4, borderline accept)

Thank you for the most complete reading of the paper we received — your summary of what the ablation does and does not establish is exactly right, and we adopt your framing in the revision.

## Q1. How should the lower bound be interpreted relative to CalFuse?

Your reading is correct and we will make it the paper's official one. Precisely:

- Theorem 1 lower-bounds the worst-cell sample complexity at Θ(α⁻² log K) for the class F_n of post-hoc calibrators whose per-cell output depends on calibration data only through per-cell empirical means. This matches HKRR's upper bound, so within F_n the analysis is tight: the per-cell route is *optimally* sample-hungry, and no cleverness inside that class escapes the cost.
- The theorem therefore functions as a design justification — the reason CalFuse pools strength across cells via per-stratum isotonic structure instead of auditing cells independently — and not as a performance guarantee for CalFuse. CalFuse's direct guarantees are of a different type: distribution-free per-stratum coverage of the Venn–Abers envelope under exchangeability, and multicalibration preservation of the parametric fusion stage under conditional independence.
- What is missing, and will now be stated as an open problem in the Limitations: a matching *upper* bound on worst-subgroup calibration error for stratified monotone calibration, which would close the loop by showing the escape from F_n buys the predicted improvement in rates rather than merely avoiding the lower bound's hypothesis. We will retitle the theory section "Why per-cell post-hoc calibration cannot suffice" so no reader mistakes the bound for a guarantee about CalFuse.

## Q2. What can and cannot be concluded from the hallucination experiment?

We will substantially sharpen this. First, transparency: the released artifacts contain the identical 5-seed protocol on five subsets (nfcorpus, scifact, fiqa, arguana, scidocs). At full coverage, CalFuse-P vs the strongest learned baseline (Linear-Learned) on hallucination-among-non-refused: scifact −3.3pp (p=0.045, d=−1.29); the other four subsets are within seed noise in both directions (e.g. fiqa −1.2pp, nfcorpus +2.2pp, all n.s.). The revision will show all five.

What can be concluded: on a subset where top-k composition is contested, improving subgroup calibration changes *which* passages enter the context, and the direct mechanism test (rank-of-first-positive distributions statistically unchanged across methods; top-k composition changed) rules out ranking sharpening — so the reduction is attributable to distractor swap, which is the causal pathway a calibration method could honestly claim. What cannot be concluded: a general "calibration reduces hallucination" effect across corpora, models, or scales. We view the experiment as an existence-plus-mechanism demonstration and will label it as such, with model scale (Qwen-2.5-7B only) listed as an explicit external-validity limit.

## Q3. Selective-NDCG on denser-relevance data

Agreed that trec-covid with n_q=15 test queries carries the signature but not much evidential weight. Two honest notes and one addition: (a) the signature's derivation predicts it should appear only under dense graded relevance, which is why the sparse-relevance subsets are uninformative rather than contradictory — we will state this prediction *before* the result so it reads as a test, not a post-hoc selection; (b) we will add bootstrap CIs over queries to make the fragility visible; (c) within the BEIR suite, dense-judgment subsets are essentially trec-covid (and to a lesser degree nfcorpus, which we will add as a secondary check); extending to TREC DL 2019–2021 passage tracks, which have the required judgment density, is the right test and we will do it for the camera-ready if accepted, or list it as the designated follow-up otherwise.

## On the main weakness: the operative piece is stratified monotone calibration, not the envelope

We accept this characterization — the family ablation shows CalFuse statistically tied with Subgroup-Isotonic on the point estimate on all seven subsets, and the envelope-based abstention experiments show no reliable advantage over point-estimate ranking (envelope-lower-bound ordering helps only at very low coverage on some subsets, within noise). The revision will (i) state in the contributions that the point-estimate gains are attributable to *stratified nonparametric monotone* calibration, (ii) reposition IVAP's contribution as the finite-sample validity guarantee and the per-prediction width — properties isotonic alone lacks and which cost nothing extra at inference (both calibrators are a sorted-array lookup) — and (iii) move the envelope-abstention comparison into the main ablation table so the tie is visible rather than implied. We believe the honest version of the claim is still a contribution: the guarantee is free given the mechanism that already wins on ECE.

We will also follow your suggestion to demote the secondary evaluation threads (selective-NDCG, envelope abstention) to clearly-marked supporting evidence so the central calibration story is not diluted.
