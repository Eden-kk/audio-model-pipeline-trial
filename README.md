# audio-model-pipeline-trial

Web playground + pipeline composer + corpus builder for **audio-model evaluation** — drop in any ASR / TTS / speaker-verification adapter, compose pipelines, run on live mic or uploaded clips, visualise per-stage I/O + latency + cost, grow a labelled corpus over time.

**Why:** AR-glasses voice-assistant stacks are a moving target. Cloud APIs, edge models, omni vs stitched, streaming vs batch, open-weight vs closed — all evolving on different cadences. This app is the harness that lets a small research team keep up without a one-off script per candidate.

**Status:** P0 MVP — audio only. Slice 0 (scaffold + Playground page) and Slice 1 (10 new adapters + AMD-ROCm deployment infra) shipped. See [`docs/PLAN.md`](docs/PLAN.md) for the full capability matrix and phasing.

---

## What's in the model library today

12 adapters, mixing cloud APIs with self-hosted CPU/GPU options. **Two require no key** (faster-whisper and Resemblyzer) — the app demos out of the box.

| # | Adapter | Category | Hosting | Env var |
|---|---------|---------|---------|---------|
| 1 | Deepgram Nova-3 | ASR | cloud | `DEEPGRAM_API_KEY` |
| 2 | Gladia | ASR | cloud | `GLADIA_API_KEY` |
| 3 | AssemblyAI Universal-2 | ASR | cloud | `ASSEMBLYAI_API_KEY` |
| 4 | Speechmatics | ASR | cloud | `SPEECHMATICS_API_KEY` |
| 5 | Groq Whisper-Turbo (LPU) | ASR | cloud | `GROQ_API_KEY` |
| 6 | **faster-whisper** *(small.en / large-v3 / large-v3-turbo / distil / turbo-en)* | ASR | self-host CPU/GPU (in-process) | — |
| 7 | **Parakeet-TDT-1.1B** | ASR | self-host (model-server, AMD-ROCm) | `MODEL_SERVER_URL` |
| 8 | **Canary-1B-flash** *(EN/DE/ES/FR)* | ASR | self-host (model-server) | `MODEL_SERVER_URL` |
| 9 | **Canary-Qwen-2.5B** | ASR | self-host (model-server) | `MODEL_SERVER_URL` |
| 10 | Cartesia Sonic-3 | TTS | cloud | `CARTESIA_API_KEY` |
| 11 | pyannote/embedding | Speaker verify | self-host CPU | `HF_TOKEN` (license at [pyannote/embedding](https://huggingface.co/pyannote/embedding)) |
| 12 | Resemblyzer | Speaker verify | self-host CPU | — |

---

## Deployment — one URL on your AMD remote machine

This is the canonical path: a single `docker compose` stack that you run on an AMD-ROCm box and reach over the internet via one URL.

```
[your laptop] ─ HTTPS ─▶ [Caddy:443] ─▶ [trial-app:8000] ──▶ [model-server:9100]
                                              │                ├ parakeet-tdt-1.1b
                                              │                ├ canary-1b-flash
                                              ▼                └ canary-qwen-2.5b
                                       SPA static
                                        (frontend/dist baked
                                         into trial-app image)
```

```bash
git clone https://github.com/Eden-kk/audio-model-pipeline-trial.git
cd audio-model-pipeline-trial
cp .env.example .env && $EDITOR .env       # PUBLIC_HOST, LETSENCRYPT_EMAIL, API keys
docker compose up -d --build               # Caddy auto-provisions TLS
```

Open `https://${PUBLIC_HOST}` from anywhere → land on the Playground.

**Full walkthrough** — including hardware matrix (MI300X / 7900 XTX / etc.), the `HSA_OVERRIDE_GFX_VERSION` table for non-MI300 GPUs, troubleshooting, and URL-exposure variants (Cloudflare Tunnel, SSH tunnel, on-prem) — lives in [`docs/AMD-DEPLOY.md`](docs/AMD-DEPLOY.md).

> **No AMD GPU?** The cloud adapters all still work — you can run the trial-app + frontend on any CPU machine. Skip the `model-server` service in compose (or run it CPU-only and accept a 5–20× slowdown for self-host models). The deploy walkthrough covers this case.

---

## Local development (no Docker)

```bash
# Backend — Python 3.11+
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in any API keys you have
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend — Vite dev server (separate terminal)
cd ../frontend
pnpm install     # or: npm install
pnpm dev         # http://localhost:5173 → talks to localhost:8000
```

The dev-server CORS allowlist already includes `localhost:5173`. To point the SPA at a non-localhost backend, set `VITE_API_URL=https://your-host` in `frontend/.env`.

---

## Architecture (one screen)

```
React + Vite + TS  ─── HTTP REST ───▶  FastAPI on uvicorn
   sidebar:                              ├── /api/adapters
     Playground                          ├── /api/pipelines  (Slice 2)
     Pipelines                           ├── /api/clips
     Run                                 ├── /api/runs
     Corpus            ◀── WebSocket ─── └── /ws/run/{id}     (per-stage events)
     Settings                                    │
                                                 ▼
                                       backend/adapters/registry
                                       — 12 adapters today
                                       — protocol-typed inputs/outputs
                                       — talks to model-server for self-host

                                                 ▼ (NeMo only)
                                       model-server FastAPI
                                       (own ROCm container, GPU-passthrough)
```

Full data model and phasing in [`docs/PLAN.md`](docs/PLAN.md).

---

## Repo layout

```
backend/             FastAPI app, adapter registry (12 adapters), runner, clip storage
frontend/            Vite + React + TS — Playground today; Pipelines/Run/Compare/Corpus in P1
model-server/        ROCm container hosting NeMo ASR (Parakeet, Canary-1B, Canary-Qwen)
docs/
  PLAN.md            canonical product plan (P0 / P1 / P2)
  AMD-DEPLOY.md      step-by-step AMD-ROCm deployment walkthrough
docker-compose.yml   3-service stack: trial-app + model-server + Caddy
Caddyfile            reverse proxy + auto-TLS
.env.example         deployment env (PUBLIC_HOST, API keys, GPU overrides)
```

Each new feature is its own commit on `main` (or a feature branch + PR).

---

## License

MIT — see `LICENSE`.
