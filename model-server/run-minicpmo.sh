#!/usr/bin/env bash
# Native CUDA launcher for MiniCPM-o-4.5 (realtime omni).
#
# Counterpart to model-server/modal_app_minicpm.py (Modal A100 lane).
# Same wire shape (POST /v1/omni, POST /v1/omni-stream, GET /health) so
# backend/adapters/minicpm_o_adapter.py routes here with only an env flip:
#   MINICPM_O_REALTIME_URL=http://localhost:9101
#
# Required env:
#   MODEL_CACHE_DIR        — roomy mount for the HF cache (the model is
#                            ~18 GB, must NOT live on /home if /home is small).
# Optional env:
#   HF_TOKEN               — Currently MiniCPM-o-4.5 is public; set if it ever
#                            becomes gated.
#   CUDA_VISIBLE_DEVICES   — pin a specific GPU index. Default 7 on this
#                            shared box (GPUs 0–6 are NeMo / vLLM / others).
#   MINICPMO_PORT          — port to bind on. Default 9101 (avoids 9100/NeMo,
#                            8001/vllm, 8000/trial-app).
#   VENV_DIR               — model-server venv. Default
#                            /raid/yid042/audio-stack/venv-minicpmo.
#   MINICPMO_MODEL         — HF model id. Default openbmb/MiniCPM-o-4_5.

set -euo pipefail

VENV_DIR="${VENV_DIR:-/raid/yid042/audio-stack/venv-minicpmo}"
MINICPMO_PORT="${MINICPMO_PORT:-9101}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
MINICPMO_MODEL="${MINICPMO_MODEL:-openbmb/MiniCPM-o-4_5}"

if [[ -z "${MODEL_CACHE_DIR:-}" ]]; then
  echo "ERROR: MODEL_CACHE_DIR must be set (e.g. /raid/yid042/audio-trial-models)" >&2
  exit 2
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: minicpmo venv missing at $VENV_DIR — bootstrap with bootstrap-minicpmo.sh" >&2
  exit 2
fi

mkdir -p "$MODEL_CACHE_DIR/hf_cache" "$MODEL_CACHE_DIR/minicpmo_cache"

export HF_HOME="$MODEL_CACHE_DIR/hf_cache"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES
export MINICPMO_MODEL

cd "$(dirname "$0")"

echo "minicpmo: venv=$VENV_DIR  model=$MINICPMO_MODEL  cache=$MODEL_CACHE_DIR  gpu=$CUDA_VISIBLE_DEVICES  port=$MINICPMO_PORT"

exec "$VENV_DIR/bin/python" -m uvicorn server_minicpmo:app \
  --host 0.0.0.0 --port "$MINICPMO_PORT" \
  --log-level info
