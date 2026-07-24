#!/usr/bin/env bash
# Turnkey GPU scaling run for the rebuttal, designed for a Lambda cloud
# instance (1x A100 or H100 recommended; A10 works, ~3x slower).
#
# From your laptop:
#   ssh ubuntu@<lambda-ip>
#   git clone https://github.com/nihalgunu/CalFuse.git && cd CalFuse
#   git checkout claude/retrieval-calibration-gap-j7oxpe
#   tmux new -s rebuttal
#   bash rebuttal/lambda_gpu_run.sh all      # or: phase1 | phase2
#
# Results are committed to branch rebuttal-gpu-results and pushed (uses
# whatever git credentials the instance has; set GIT_PUSH=0 to skip and
# scp eval/ back instead).
#
# Phase 1  — statistical power: 5 extra seeds (2031-2035) for the existing
#            Qwen-2.5-7B protocol, 5 subsets x 50 queries. Doubles the seed
#            count of every hallucination comparison to n=10.
#            Est: ~9-15 h on A100/H100, ~30 h on A10.
# Phase 2  — model transfer (answers utWJ Q4 "does it transfer to bigger
#            models?"): Qwen-2.5-14B-Instruct, original 5 seeds, on the two
#            subsets where the 7B effect is present/neutral (scifact, fiqa)
#            plus nfcorpus as the adversarial case. Output isolated in
#            eval/multiseed_qwen14b/. Est: ~8-12 h on A100/H100.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PHASE="${1:-all}"
GIT_PUSH="${GIT_PUSH:-1}"
RESULTS_BRANCH="rebuttal-gpu-results"
NEW_SEEDS=(2031 2032 2033 2034 2035)
ORIG_SEEDS=(2026 2027 2028 2029 2030)

echo "== environment =="
nvidia-smi --query-gpu=name,memory.total --format=csv || {
  echo "No GPU visible - aborting"; exit 1; }
python3 -m pip install -e ".[full]" --quiet

# The multiseed runner reads/writes eval/llm_hallu_<subset>_seed<seed>.json.
# Released 5-seed verdicts live in eval/multiseed/ - link them in as cache so
# only genuinely new (subset, seed) cells run.
mkdir -p eval/multiseed
for f in eval/multiseed/llm_hallu_*.json; do
  [ -e "$f" ] && ln -sf "multiseed/$(basename "$f")" "eval/$(basename "$f")"
done

run_phase1() {
  echo "== Phase 1: +5 seeds, Qwen-2.5-7B, 5 subsets =="
  PYTHONPATH=. python3 scripts/multiseed_llm_hallu.py \
    --seeds "${NEW_SEEDS[@]}" \
    --subsets nfcorpus scifact fiqa arguana scidocs \
    --out eval/llm_hallu_multiseed_p1.json
  # Collect the new per-seed verdict files into eval/multiseed/.
  for s in "${NEW_SEEDS[@]}"; do
    for f in eval/llm_hallu_*_seed"$s".json; do
      [ -L "$f" ] && continue
      [ -e "$f" ] && mv "$f" "eval/multiseed/$(basename "$f")"
    done
  done
}

run_phase2() {
  echo "== Phase 2: Qwen-2.5-14B model transfer =="
  mkdir -p eval/multiseed_qwen14b
  for subset in scifact fiqa nfcorpus; do
    for seed in "${ORIG_SEEDS[@]}"; do
      out="eval/multiseed_qwen14b/llm_hallu_${subset}_seed${seed}.json"
      [ -e "$out" ] && { echo "  cache hit: $out"; continue; }
      PYTHONPATH=. PYTHONHASHSEED=0 python3 scripts/llm_hallucination_eval.py \
        --subset "$subset" --model Qwen/Qwen2.5-14B-Instruct \
        --n-queries 50 --top-k 5 --seed "$seed" \
        --methods bm25_platt bge_platt linear_learned calfuse_parametric calfuse_conformal \
        --out "$out"
    done
  done
}

case "$PHASE" in
  phase1) run_phase1 ;;
  phase2) run_phase2 ;;
  all)    run_phase1; run_phase2 ;;
  *) echo "usage: $0 [phase1|phase2|all]"; exit 1 ;;
esac

if [ "$GIT_PUSH" = "1" ]; then
  echo "== pushing results to $RESULTS_BRANCH =="
  git checkout -B "$RESULTS_BRANCH"
  git add -f eval/multiseed/ eval/multiseed_qwen14b/ 2>/dev/null || true
  git commit -m "GPU rebuttal results: extra seeds + Qwen-14B transfer" || echo "nothing new to commit"
  git push -u origin "$RESULTS_BRANCH"
else
  echo "GIT_PUSH=0: results left in eval/multiseed/ and eval/multiseed_qwen14b/"
fi
echo "Done."
