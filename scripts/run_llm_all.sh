#!/bin/bash
# Zero-shot LLM entity matching with a local 8B model (MLX), replaying the MatchGPT protocol.
# Prompt: domain-complex-force, the variant that works best for most models in Peeters et al.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
mkdir -p results/llm

for ds in abt-buy-sampled-gs walmart-amazon-sampled-gs wdcproducts-80cc-seen-sampled-250-gs-2 historical50k-person-hard; do
  echo "=== $ds $(date +%T)"
  nice $PY scripts/llm_zeroshot_mlx.py --model "$MODEL" --prompt domain-complex-force \
      --datasets "$ds" >> results/llm/run.log 2>&1
  tail -1 results/llm/run.log
done
