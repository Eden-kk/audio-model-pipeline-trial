#!/usr/bin/env bash
# Native CUDA launcher for the model-server (NVIDIA B200 / H100 / A100).
#
# Counterpart to model-server/Dockerfile (AMD-ROCm) and modal_app.py (Modal).
# Use this when docker is unavailable on the host (e.g., user not in `docker`
# group, no sudo, no rootless docker).
#
# Required env (typically from the repo's .env, or your shell):
#   MODEL_CACHE_DIR        — roomy mount for HF + NeMo weights (must NOT be
#                            under $HOME if /home is small). Required.
#   HF_TOKEN               — needed for gated models (Canary). Optional.
#   CUDA_VISIBLE_DEVICES   — pin a specific GPU index. Default 0.
#   MODEL_SERVER_PORT      — port to bind on. Default 9100.
#   VENV_DIR               — path to the model-server venv. Default
#                            /raid/yid042/audio-stack/venv-modelserver.
#
# After this script is running, point the trial-app at it via:
#   export MODEL_SERVER_BACKEND=local
#   export MODEL_SERVER_LOCAL_URL=http://localhost:${MODEL_SERVER_PORT:-9100}

set -euo pipefail

VENV_DIR="${VENV_DIR:-/raid/yid042/audio-stack/venv-modelserver}"
MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-9100}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "${MODEL_CACHE_DIR:-}" ]]; then
  echo "ERROR: MODEL_CACHE_DIR must be set (e.g. /raid/yid042/audio-trial-models)" >&2
  exit 2
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv missing at $VENV_DIR — bootstrap with the install commands" >&2
  echo "       documented in README.md → 'Local CUDA (native, no docker)'." >&2
  exit 2
fi

mkdir -p "$MODEL_CACHE_DIR/hf_cache" "$MODEL_CACHE_DIR/nemo_cache"

export HF_HOME="$MODEL_CACHE_DIR/hf_cache"
export NEMO_CACHE_DIR="$MODEL_CACHE_DIR/nemo_cache"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES

cd "$(dirname "$0")"

echo "model-server (CUDA): venv=$VENV_DIR  cache=$MODEL_CACHE_DIR  gpu=$CUDA_VISIBLE_DEVICES  port=$MODEL_SERVER_PORT"

exec "$VENV_DIR/bin/python" -m uvicorn server:app \
  --host 0.0.0.0 --port "$MODEL_SERVER_PORT" \
  --log-level info
