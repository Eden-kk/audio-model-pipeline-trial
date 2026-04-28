#!/usr/bin/env bash
# Native vLLM launcher for the qwen_intent adapter.
#
# Hosts an OpenAI-compatible /v1 endpoint that backend/adapters/qwen_intent_adapter.py
# already knows how to talk to via INTENT_LLM_URL — the adapter never had a
# Modal-specific code path; only the URL it points at changes.
#
# Required env:
#   MODEL_CACHE_DIR        — roomy mount; vLLM's HF cache lives at
#                            $MODEL_CACHE_DIR/vllm. Required.
#   HF_TOKEN               — needed if the chosen model is gated. Optional.
#   VLLM_GPU               — GPU index to pin. Default 1 (so we don't fight
#                            run-cuda.sh which defaults to GPU 0).
#   VLLM_MODEL             — HF model id. Default Qwen/Qwen2.5-7B-Instruct
#                            (matches qwen_intent_adapter._DEFAULT_MODEL).
#   VLLM_PORT              — port to bind on. Default 8001 (avoids the
#                            trial-app's 8000 and model-server's 9100).
#   VLLM_MAX_MODEL_LEN     — context window. Default 8192.
#   VENV_DIR               — vllm venv path. Default
#                            /raid/yid042/audio-stack/venv-vllm.
#
# After this script is running, point qwen_intent at it via:
#   export INTENT_LLM_URL=http://localhost:${VLLM_PORT:-8001}/v1

set -euo pipefail

VENV_DIR="${VENV_DIR:-/raid/yid042/audio-stack/venv-vllm}"
VLLM_GPU="${VLLM_GPU:-1}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

if [[ -z "${MODEL_CACHE_DIR:-}" ]]; then
  echo "ERROR: MODEL_CACHE_DIR must be set (e.g. /raid/yid042/audio-trial-models)" >&2
  exit 2
fi
if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
  echo "ERROR: vLLM venv missing at $VENV_DIR — bootstrap with bootstrap-vllm.sh" >&2
  exit 2
fi

mkdir -p "$MODEL_CACHE_DIR/vllm"

export HF_HOME="$MODEL_CACHE_DIR/vllm"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$VLLM_GPU"

echo "vllm: model=$VLLM_MODEL  gpu=$VLLM_GPU  port=$VLLM_PORT  cache=$MODEL_CACHE_DIR/vllm"

exec "$VENV_DIR/bin/vllm" serve "$VLLM_MODEL" \
  --host 0.0.0.0 --port "$VLLM_PORT" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --served-model-name "$VLLM_MODEL"
