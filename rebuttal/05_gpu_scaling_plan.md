# GPU scaling plan (Lambda) — what to run, why, and what it buys

The sandbox this rebuttal was prepared in has no GPU and no Lambda
credentials, so the GPU-bound experiments must run on your Lambda instance.
`rebuttal/lambda_gpu_run.sh` is turnkey: clone the repo on the box, check out
this branch, run it inside tmux, and results come back on branch
`rebuttal-gpu-results` for analysis here.

## What is worth GPU time, ranked

### Phase 1 — double the seeds (n=5 → n=10) on the existing protocol
Qwen-2.5-7B, seeds 2031–2035, five subsets × 50 queries × 5 methods.

*What it buys:* every hallucination comparison gets twice the seeds. If the
scifact effect (−3.31 pp, p=0.045 at n=5) is real, p should tighten toward
~0.01 and the rebuttal can say "10 seeds"; if fiqa's −1.2 pp is real it may
reach significance. *Risk accepted:* if scifact drifts above 0.05 at n=10, we
report that — better we learn it than a replicator.

*Cost:* ~9–15 h on one A100/H100 (~$15–40), ~30 h on A10.

### Phase 2 — model transfer (the question only GPUs can answer)
Qwen-2.5-14B-Instruct, original 5 seeds, on scifact + fiqa (effect
present/neutral at 7B) and nfcorpus (adversarial case, fusion worse than
BM25-only at 7B). Output isolated in `eval/multiseed_qwen14b/`.

*What it buys:* Reviewer utWJ asks verbatim "does it transfer to bigger
models or other tasks?" — this is the only question in any review that no
frozen-artifact analysis can touch. Even a directional replication on
scifact at 14B upgrades the claim from "one model" to "two model scales";
a null is a clean scoped limitation we report.

*Cost:* ~8–12 h on one A100/H100 (~$20–35).

### Not worth the window (say so in the rebuttal instead)
- **TREC-DL selective-NDCG**: needs full dense-encoder score matrices on a
  new corpus (many GPU-hours) *and* new qrels plumbing — a camera-ready
  commitment, already promised to T3MF.
- **More queries per subset** (50 → 100+): scifact has only 90 test queries;
  power grows faster with seeds than queries here (between-seed variance
  dominates), and Phase 1 is the cheaper axis.
- **A third model family** (Llama/Mistral): nice-to-have; Phase 2 already
  converts "single model" into "model-scale comparison". Only if the
  instance is idle anyway.

## How results flow back

1. Lambda box pushes `rebuttal-gpu-results` (verdict JSONs only).
2. Here: `git fetch origin rebuttal-gpu-results && git checkout
   rebuttal-gpu-results -- eval/multiseed eval/multiseed_qwen14b`, then
   rerun `rebuttal/run_rebuttal_experiments.py` (E1 picks up all seeds it
   finds) plus a 7B-vs-14B comparison; drafts get updated with n=10 /
   two-scale numbers.

## Decision guidance

If the discussion deadline is close, **Phase 1 alone is enough** to defuse
"statistically thin" (it's the exact protocol reviewers saw, just doubled).
Phase 2 is the highest-value *new* evidence per GPU-hour because it answers
utWJ's explicit question. Run `all` if the box can run ~24 h before the
response deadline.

## If you want this driven from the sandbox

Provide a Lambda Cloud API key (add `LAMBDA_API_KEY` to the session
environment or paste it) and, network policy permitting, instance launch +
remote execution can be orchestrated from here. Without credentials, the
one-command path above is the fastest route.
