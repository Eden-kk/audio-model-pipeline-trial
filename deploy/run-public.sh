#!/usr/bin/env bash
# Path D — full local stack orchestrator + public URL via Cloudflare Tunnel.
#
# Brings up all four processes that make the trial-app reachable from the
# internet on a NVIDIA host where docker is unavailable:
#
#   1. model-server (CUDA, NeMo) — model-server/run-cuda.sh on $MODEL_SERVER_PORT
#   2. vLLM Qwen-Intent          — model-server/run-vllm.sh on $VLLM_PORT
#   3. trial-app (FastAPI + SPA) — uvicorn on :8000
#   4. cloudflared (ephemeral)   — prints a *.trycloudflare.com URL
#
# Idempotent: if a component's PID file exists and the process is alive, we
# skip restarting it. To force a clean reboot:  bash deploy/run-public.sh stop
# then re-run with no args.
#
# Required env (typically in repo .env):
#   MODEL_CACHE_DIR   — roomy mount for HF + NeMo + vLLM caches.
# Optional:
#   HF_TOKEN          — gated models (Canary, etc.).
#   AUDIO_STACK_DIR   — venvs root. Default /raid/yid042/audio-stack.
#   MODEL_SERVER_PORT — model-server port. Default 9100.
#   VLLM_PORT         — vLLM port. Default 8001.
#   TRIAL_APP_PORT    — trial-app port. Default 8000.
#
# Notes:
#   - Frontend is built with `pnpm exec vite build` (skipping `tsc -b`) so
#     pre-existing strict-TS warnings don't block deploy. Production-grade
#     deploys should clean those up; for evaluation infra (this repo) the
#     bundle ships fine.
#   - cloudflared is the binary downloaded to $AUDIO_STACK_DIR/bin during
#     bootstrap (see slice-0 in the migration plan), NOT a docker image.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUDIO_STACK_DIR="${AUDIO_STACK_DIR:-/raid/yid042/audio-stack}"
LOG_DIR="$AUDIO_STACK_DIR/logs"
RUN_DIR="$AUDIO_STACK_DIR/run"
BIN_DIR="$AUDIO_STACK_DIR/bin"

MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-9100}"
VLLM_PORT="${VLLM_PORT:-8001}"
TRIAL_APP_PORT="${TRIAL_APP_PORT:-8000}"

# MiniCPM-o realtime omni (optional; opt-in via ENABLE_MINICPMO=1).
# Default OFF so a fresh clone on a smaller box doesn't OOM. On the B200
# dev host, set ENABLE_MINICPMO=1 in your local .env (or shell rc).
ENABLE_MINICPMO="${ENABLE_MINICPMO:-0}"
MINICPMO_PORT="${MINICPMO_PORT:-9101}"
MINICPMO_GPU="${MINICPMO_GPU:-7}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

# ─── Helpers ───────────────────────────────────────────────────────────────

is_alive() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null
}

start_bg() {
  local name="$1"; shift
  local pidfile="$RUN_DIR/$name.pid"
  local logfile="$LOG_DIR/$name.log"
  if is_alive "$pidfile"; then
    echo "  [skip] $name already alive (pid $(cat "$pidfile"))"
    return 0
  fi
  echo "  [boot] $name → $logfile"
  nohup "$@" > "$logfile" 2>&1 &
  echo $! > "$pidfile"
  disown $!
}

stop_all() {
  # Order: kill outward-facing pieces first (cloudflared, trial-app) so
  # the heavy GPU processes don't see ECONNRESET storms while shutting down.
  for name in cloudflared trial-app minicpmo vllm model-server; do
    local pidfile="$RUN_DIR/$name.pid"
    if is_alive "$pidfile"; then
      local pid; pid=$(cat "$pidfile")
      echo "  [stop] $name (pid $pid)"
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  done
}

case "${1:-up}" in
  stop|down)
    stop_all
    exit 0
    ;;
  status)
    for name in model-server vllm minicpmo trial-app cloudflared; do
      if is_alive "$RUN_DIR/$name.pid"; then
        echo "  $name: alive (pid $(cat "$RUN_DIR/$name.pid"))"
      else
        echo "  $name: down"
      fi
    done
    exit 0
    ;;
  url)
    # Public URLs assigned by cloudflared look like
    #   https://<adjective>-<noun>-<noun>-<digits>-<word>.trycloudflare.com
    # Always contain at least 2 dashes in the subdomain. The API endpoint
    # (api.trycloudflare.com) appears in the registration banner — exclude it.
    grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/cloudflared.log" 2>/dev/null \
      | grep -v '^https://api\.trycloudflare\.com$' \
      | tail -1 || { echo "no URL yet — check $LOG_DIR/cloudflared.log" >&2; exit 1; }
    exit 0
    ;;
  up|"")
    ;;
  *)
    echo "usage: $0 {up|stop|status|url}" >&2
    exit 2
    ;;
esac

if [[ -z "${MODEL_CACHE_DIR:-}" ]]; then
  echo "ERROR: MODEL_CACHE_DIR must be set (e.g. /raid/yid042/audio-trial-models)" >&2
  exit 2
fi

# ─── Build frontend (idempotent — vite-only, ~1s on cache) ────────────────

if [[ ! -f "$REPO_ROOT/frontend/dist/index.html" ]]; then
  echo "  [build] frontend SPA"
  if command -v pnpm >/dev/null 2>&1; then
    (cd "$REPO_ROOT/frontend" && pnpm install --frozen-lockfile && VITE_API_URL="" pnpm exec vite build)
  else
    (cd "$REPO_ROOT/frontend" && npm ci && VITE_API_URL="" npx vite build)
  fi
fi

# ─── Component 1: NeMo model-server ────────────────────────────────────────

start_bg model-server env \
  MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
  CUDA_VISIBLE_DEVICES="${MODEL_SERVER_GPU:-0}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  MODEL_SERVER_PORT="$MODEL_SERVER_PORT" \
  bash "$REPO_ROOT/model-server/run-cuda.sh"

# ─── Component 2: vLLM Qwen-Intent ─────────────────────────────────────────

start_bg vllm env \
  MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
  HF_TOKEN="${HF_TOKEN:-}" \
  VLLM_GPU="${VLLM_GPU:-1}" \
  VLLM_PORT="$VLLM_PORT" \
  bash "$REPO_ROOT/model-server/run-vllm.sh"

# ─── Component 2.5: MiniCPM-o realtime omni (opt-in) ───────────────────────
# Off by default so contributors on smaller boxes don't auto-OOM. Enable
# via ENABLE_MINICPMO=1 in your local .env. Cold-start ≈ 60–120s on first
# run (downloads ~18 GB into $MODEL_CACHE_DIR/hf_cache), 30–40s thereafter.

if [[ "$ENABLE_MINICPMO" == "1" ]]; then
  start_bg minicpmo env \
    MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
    HF_TOKEN="${HF_TOKEN:-}" \
    CUDA_VISIBLE_DEVICES="$MINICPMO_GPU" \
    MINICPMO_PORT="$MINICPMO_PORT" \
    bash "$REPO_ROOT/model-server/run-minicpmo.sh"
fi

# ─── Component 3: trial-app FastAPI ────────────────────────────────────────

TRIAL_APP_VENV="$AUDIO_STACK_DIR/venv-trialapp"
if [[ ! -x "$TRIAL_APP_VENV/bin/uvicorn" ]]; then
  echo "ERROR: trial-app venv missing at $TRIAL_APP_VENV — bootstrap with:" >&2
  echo "  uv venv --python 3.12 $TRIAL_APP_VENV" >&2
  echo "  $AUDIO_STACK_DIR/bin/uv pip install --python $TRIAL_APP_VENV/bin/python -r $REPO_ROOT/backend/requirements.txt" >&2
  exit 2
fi

# If MiniCPM-o is enabled, point the trial-app's adapter at our local
# upstream; otherwise leave the env var empty so the adapter reports
# "unset" in /api/settings instead of hammering a dead host.
if [[ "$ENABLE_MINICPMO" == "1" ]]; then
  MINICPM_O_REALTIME_URL_VAL="http://localhost:$MINICPMO_PORT"
else
  MINICPM_O_REALTIME_URL_VAL="${MINICPM_O_REALTIME_URL:-}"
fi

start_bg trial-app env \
  MODEL_SERVER_BACKEND=local \
  MODEL_SERVER_LOCAL_URL="http://localhost:$MODEL_SERVER_PORT" \
  INTENT_LLM_URL="http://localhost:$VLLM_PORT/v1" \
  INTENT_LLM_MODEL="${INTENT_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
  MINICPM_O_REALTIME_URL="$MINICPM_O_REALTIME_URL_VAL" \
  HF_TOKEN="${HF_TOKEN:-}" \
  FRONTEND_DIST="$REPO_ROOT/frontend/dist" \
  DATA_DIR="${DATA_DIR:-$REPO_ROOT/backend/data}" \
  PYTHONUNBUFFERED=1 \
  bash -c "cd $REPO_ROOT/backend && exec $TRIAL_APP_VENV/bin/python -m uvicorn main:app --host 0.0.0.0 --port $TRIAL_APP_PORT"

# ─── Component 4: cloudflared (ephemeral tunnel) ───────────────────────────

CLOUDFLARED="${CLOUDFLARED:-$BIN_DIR/cloudflared}"
if [[ ! -x "$CLOUDFLARED" ]]; then
  echo "ERROR: cloudflared binary not found at $CLOUDFLARED" >&2
  echo "  Download with: curl -fL --output $CLOUDFLARED \\" >&2
  echo "    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" >&2
  echo "  chmod +x $CLOUDFLARED" >&2
  exit 2
fi

# --protocol http2 forces HTTP/2 over TCP (port 443) instead of QUIC (UDP 7844).
# UCSD egress and many corporate networks block outbound UDP 7844.
# --edge-ip-version 4 avoids IPv6 timeouts on hosts where v6 routes to
# Cloudflare's edge are flaky (we hit "api.trycloudflare.com context
# deadline exceeded" otherwise).
start_bg cloudflared "$CLOUDFLARED" tunnel --no-autoupdate \
  --protocol http2 \
  --edge-ip-version 4 \
  --url "http://localhost:$TRIAL_APP_PORT"

echo
echo "Stack booting. Tail logs at $LOG_DIR/. To get the public URL after"
echo "cloudflared registers (~5–10 s):"
echo "  bash $0 url"
