#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server hosting the merged SFT+DPO model.
#
# Run from a separate venv that has vllm installed (since vllm pins torch
# tightly and we don't want to disturb the training env).
#
# Usage:
#   bash serve/vllm_serve.sh [model_path] [port]
#
# Default: serves /workspace/merged/dpo on port 8000.

set -euo pipefail

MODEL_PATH="${1:-/workspace/merged/dpo}"
PORT="${2:-8000}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model dir not found: $MODEL_PATH"
    echo "Run scripts/merge_lora.py first to merge LoRA into base."
    exit 1
fi

echo "Serving $MODEL_PATH on port $PORT"
exec vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --served-model-name qwen3-dpo \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85
