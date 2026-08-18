#!/usr/bin/env bash
# Serve a model with vLLM for QoldaEvalKit. See README.md.
#
#   ./serve_model.sh
#   MODEL_PATH=/path/to/qolda-avl-5b PORT=8008 ./serve_model.sh
set -euo pipefail

# ================================ CONFIG =====================================

MODEL_PATH="${MODEL_PATH:-/path_to_model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"          # keep >= CONCURRENCY in run_eval.sh

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Multimodal models only. Empty to omit.
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"audio\": 4, \"image\": 4}}"

# qwen3 | deepseek_r1 | granite | glm45 | gptoss. Empty for non-reasoning models.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Passed straight to vllm, e.g. "--enable-prefix-caching --swap-space 8".
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ============================== END CONFIG ===================================

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

args=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --host "$HOST"
  --port "$PORT"
)
[[ -n "$LIMIT_MM_PER_PROMPT" ]] && args+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
[[ -n "$REASONING_PARSER" ]]    && args+=(--reasoning-parser "$REASONING_PARSER")

echo "serving $SERVED_MODEL_NAME on http://${HOST}:${PORT}/v1  (gpus ${CUDA_DEVICES})"

# shellcheck disable=SC2086
exec vllm serve "${args[@]}" $EXTRA_ARGS
