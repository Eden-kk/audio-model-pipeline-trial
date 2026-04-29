# audio-model-pipeline-trial

Web playground + pipeline composer + corpus builder for evaluating ASR / TTS / speaker-verification adapters against AR-glasses voice-assistant scenarios. P0 audio-only MVP. Evaluation infrastructure, not a customer product — iteration speed > polish.

## Layout
- `backend/` — Python (FastAPI), adapters, pipelines, metrics, ingest, storage. Deps in `backend/requirements.txt`.
- `frontend/` — React + Vite + TS. Deps in `frontend/package.json`.
- `model-server/` — serves Parakeet / Canary / Canary-Qwen. Three lanes share one `server.py` + `model_loader.py`:
  - `Dockerfile` + `docker-compose.yml` — AMD-ROCm container.
  - `modal_app.py` — Modal NVIDIA L4.
  - `bootstrap-cuda.sh` + `run-cuda.sh` — native CUDA venv (no docker; for hosts where you can't access the docker daemon).
- `deploy/`, `Caddyfile`, `docker-compose.yml` — deployment paths A (Modal) / B (AMD+Caddy) / C (Cloudflare Tunnel; recommended) / D (native CUDA, no docker).
- `scripts/` — corpus-building helpers.
- `docs/PLAN.md` — authoritative scope + locked decisions + phasing.
- `.env.example` — adapter API keys, ROCm gfx override, `MODEL_SERVER_BACKEND` (modal|amd|local), `MODEL_CACHE_DIR`.

## Build & run (locally on b200)
- Backend deps: `cd backend && pip install -r requirements.txt`
- Frontend deps: `cd frontend && pnpm install` (or `npm install`)
- Backend dev: `cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Frontend dev: `cd frontend && pnpm dev`
- Full stack: `docker compose up -d --build` (reads `.env` next to compose file)

## Conventions
- Two adapters (`faster-whisper`, `Resemblyzer`) need NO API key — keep "demo out of the box" working.
- New adapters land under `backend/adapters/`, expose a thin uniform interface, gate cloud calls behind env-var keys.
- AMD-ROCm hosting is path B; native NVIDIA hosting is path D; default deployment is path C (Cloudflare Tunnel + Modal model-server).
- Self-host model routing happens entirely in `backend/adapters/_nemo_http.py::model_server_url()` — switch lanes by env, not by code.
- Don't put secrets in the repo. `.env` is gitignored; `.env.example` documents what's needed.

## Dev box hardware
- `mlsys-b200.ucsd.edu`: NVIDIA B200 × 8 (CUDA 12.9, driver 570) — **not** AMD-ROCm. Use `MODEL_SERVER_BACKEND=local` + path D for self-host. The repo's AMD path is preserved for other hosts.

## See also
- @MEMORY.md — failed approaches, decisions made, open questions
- @docs/PLAN.md — full capability matrix and phasing
- @specs/ — spec-driven plans for active features (created per-feature)
