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

## Deployment

Three paths today; pick whichever matches your hardware + your DNS / port situation. The frontend + backend code is identical across all three — only the model-server location and the public-URL story change.

| Path | When                                              | Public URL via | Model-server target |
|------|---------------------------------------------------|----------------|---------------------|
| **A. Local + Modal**                  | No AMD GPU yet, or just testing | localhost / your own tunnel  | NVIDIA L4 on Modal       |
| **B. All-on-AMD with Caddy**          | AMD-ROCm box, you own a domain  | Caddy + Let's Encrypt (open 80/443) | ROCm container in same compose |
| **C. Cloudflare Tunnel** (recommended) | Public URL with no open ports / no domain required | `cloudflared` (free) | Modal **or** AMD (separate stack) |

The trial-app picks the model-server target via env (set in `.env`):

```bash
MODEL_SERVER_BACKEND=modal|amd            # which target — default modal
MODEL_SERVER_MODAL_URL=https://...        # printed by `modal deploy`
MODEL_SERVER_AMD_URL=http://localhost:9100  # AMD docker-compose host:port
# Legacy single knob still works and beats both above:
MODEL_SERVER_URL=https://...
```

### Path A — Local + Modal (no AMD needed)

```bash
# 1. Deploy the model-server to Modal (one time)
pip install modal && modal token new
cd model-server && modal deploy modal_app.py
# → https://<workspace>--audio-trial-model-server-modelserver-fastapi.modal.run

# 2. Run the trial-app locally
cd ../backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cat >> .env <<EOF
MODEL_SERVER_BACKEND=modal
MODEL_SERVER_MODAL_URL=https://<your-modal-url>
EOF
uvicorn main:app --port 8000 --reload &
cd ../frontend && pnpm install && pnpm dev   # http://localhost:5173
```

Full walkthrough + cost notes + troubleshooting in [`docs/MODAL-DEPLOY.md`](docs/MODAL-DEPLOY.md).

### Path B — All-on-AMD (Caddy + your own domain)

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

Open `https://${PUBLIC_HOST}` from anywhere → land on the Playground. Requires inbound 80/443 + a DNS record. Full walkthrough in [`docs/AMD-DEPLOY.md`](docs/AMD-DEPLOY.md).

### Path C — Cloudflare Tunnel (no open ports, no domain required)

This is the **recommended path for a remote-server deploy** where you want a public URL fast: Cloudflare Tunnel exits *outbound* from your host (so no firewall/router changes) and gives you a public HTTPS URL with TLS handled by CF.

Two flavours; pick one.

**C-1. Ephemeral (fastest — random `*.trycloudflare.com` URL, no account, no domain).**

```bash
# 1. (optional) bring up a model-server elsewhere — Modal or AMD docker-compose.
#    Set MODEL_SERVER_BACKEND + MODEL_SERVER_*_URL in .env first.

# 2. start trial-app + cloudflared
cd deploy
docker compose -f docker-compose.cloudflared.yml up -d

# 3. grab the public URL printed in cloudflared logs
docker compose -f docker-compose.cloudflared.yml logs cloudflared-ephemeral \
  | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1

# 4. open it from any machine on the internet → Playground loads
```

The URL changes on every restart, but is otherwise identical to a "real" deploy: WebSockets work, the live-mic save flow works, the public URL is reachable from this machine and any other.

**C-2. Named tunnel (persistent URL on your own Cloudflare-managed domain).**

```bash
# one-time
cloudflared tunnel login                              # browser auth
cloudflared tunnel create audio-trial-app             # prints TUNNEL_UUID
cp deploy/cloudflared/config.yml.example deploy/cloudflared/config.yml
$EDITOR deploy/cloudflared/config.yml                 # paste UUID + your hostname
cloudflared tunnel route dns audio-trial-app audio-trial.<your-domain>

# bring up the stack (named profile)
cd deploy
docker compose -f docker-compose.cloudflared.yml --profile named up -d

# verify from any machine
curl https://audio-trial.<your-domain>/api/adapters | jq '. | length'   # ≥ 12
```

> **No AMD GPU?** Path A or Path C-1 with `MODEL_SERVER_BACKEND=modal` works without any GPU on your local box.

### Capturing live audio for the AR-glass benchmark

Once the app is reachable, recording your own ground-truth corpus is one click:

1. Visit `/playground` on the public URL
2. Pick a streaming adapter (Deepgram Nova-3 or AssemblyAI Universal-Streaming)
3. Select **"Stream from mic"**, tick **"Save to corpus"**, hit Start
4. Speak, hit Stop
5. The captured PCM lands as a `live-mic` clip in the Corpus page tagged `ar-glass-capture` + `vendor-{deepgram|assemblyai}` + auto-tagger output (SNR, language, speaker count, speech ratio, duration); the streaming transcript is stored as the ground-truth seed
6. Open the clip → hand-correct `captured_transcript` if needed (`PATCH /api/clips/{id}`); see [`backend/data/README.md`](backend/data/README.md) for the recipe

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
backend/             FastAPI app, adapter registry (16+ adapters), runner, clip storage
  ingest/auto_tag.py   audio-derived auto-tagger (SNR, LID, speech ratio, speaker count)
  data/              corpus + run history (gitignored except README)
frontend/            Vite + React + TS — Playground / Pipelines / Run / Corpus / Settings
model-server/        NeMo ASR host — two deploy targets:
  Dockerfile           AMD ROCm image
  docker-compose.yml   one-command AMD deploy
  modal_app.py         Modal CUDA L4 wrapper
  model_loader.py      shared loader code
deploy/              Cloudflare Tunnel deploy stack (Path C)
  docker-compose.cloudflared.yml   trial-app + cloudflared (ephemeral or named)
  cloudflared/                     named-tunnel config example + .gitignore
docker-compose.yml   Caddy + AMD all-on-one (Path B)
Caddyfile            reverse proxy + auto-TLS
.env.example         deployment env (API keys, MODEL_SERVER_BACKEND, etc.)
docs/
  PLAN.md            canonical product plan (P0 / P1 / P2)
  AMD-DEPLOY.md      step-by-step AMD-ROCm deployment walkthrough
  MODAL-DEPLOY.md    Modal deployment walkthrough
```

Each new feature is its own commit on `main` (or a feature branch + PR).

---

## License

MIT — see `LICENSE`.
