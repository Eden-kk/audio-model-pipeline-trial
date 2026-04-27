# audio-model-pipeline-trial

Web playground + pipeline composer + corpus builder for **audio-model evaluation** — drop in any ASR / TTS / speaker-verification adapter, compose pipelines, run on live mic or uploaded clips, visualise per-stage I/O + latency + cost, grow a labelled corpus over time.

**Why:** AR-glasses voice-assistant stacks are a moving target. Cloud APIs, edge models, omni vs stitched, streaming vs batch, open-weight vs closed — all evolving on different cadences. This app is the harness that lets a small research team keep up without a one-off script per candidate.

**Status:** P0 MVP — audio-only. See [`docs/PLAN.md`](docs/PLAN.md) for the full capability matrix and phasing.

---

## Quick local run

```bash
# 1. Backend — FastAPI on Python 3.11+
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 2. Frontend — Vite + React + TS
cd ../frontend
pnpm install     # or: npm install
pnpm dev         # http://localhost:5173
```

Open `http://localhost:5173` in your browser. Sidebar exposes Playground / Pipelines / Run / Corpus / Settings.

---

## P0 adapters and required env vars

10 adapters, mixing cloud APIs with self-hosted CPU/GPU options. Two of them require **no key** (faster-whisper and Resemblyzer) — the app demos out of the box.

| # | Adapter | Category | Hosting | Env var |
|---|---------|---------|---------|---------|
| 1 | Deepgram Nova-3 | ASR | cloud | `DEEPGRAM_API_KEY` |
| 2 | Gladia | ASR | cloud | `GLADIA_API_KEY` |
| 3 | AssemblyAI Universal-2 | ASR | cloud | `ASSEMBLYAI_API_KEY` |
| 4 | Speechmatics | ASR | cloud | `SPEECHMATICS_API_KEY` |
| 5 | Groq Whisper | ASR | cloud | `GROQ_API_KEY` |
| 6 | **faster-whisper** | ASR | **self-host CPU/GPU (in-process)** | — |
| 7 | **Parakeet-TDT-1.1B** | ASR | **self-host (Modal today, AMD later)** | `PARAKEET_MODAL_URL` |
| 8 | Cartesia Sonic-3 | TTS | cloud | `CARTESIA_API_KEY` |
| 9 | pyannote/embedding | Speaker verify | self-host CPU | `HF_TOKEN` (accept license at [pyannote/embedding](https://huggingface.co/pyannote/embedding)) |
| 10 | Resemblyzer | Speaker verify | self-host CPU | — |

P1 adds OpenAI, ElevenLabs, GoogleChirp, Azure, ElevenLabs Scribe, Gemini Live (omni), gpt-realtime (omni), Moshi, Stitched pipeline, plus pyannote diarization. See `docs/PLAN.md` for the full P1 capability matrix.

---

## Remote deployment + URL exposure

You'll deploy on a remote server and want to visit the URL from a different machine. There are three coexisting paths — pick the one that matches your server.

### Path A — single Linux server with Docker (simplest)

A regular Linux box (cloud VM or on-prem) with Docker installed. The whole app — backend + frontend served by nginx — runs in one `docker compose` stack on a single port.

```bash
# On the remote server
git clone https://github.com/Eden-kk/audio-model-pipeline-trial.git
cd audio-model-pipeline-trial
cp .env.example .env && $EDITOR .env       # fill in API keys
docker compose up -d --build               # binds 0.0.0.0:8080 by default
```

To visit from your laptop, you have three sub-options:

1. **Direct port** — if the server has a public IP and port 8080 is open (or 443 with a TLS terminator):
   ```
   http://<server-public-ip>:8080
   ```
2. **Cloudflare Tunnel** (recommended; free, no port-forwarding) — `cloudflared tunnel --url http://localhost:8080` prints a `https://*.trycloudflare.com` URL.
3. **Tailscale / ZeroTier / SSH tunnel** — if the server is on a private network: `ssh -L 8080:localhost:8080 user@server` then visit `http://localhost:8080` locally.

### Path B — serverless via Modal Labs

If you'd rather not manage a server, deploy the FastAPI backend as a Modal app (`modal_app.py` is provided). Modal mints a public HTTPS URL automatically.

```bash
pip install modal
modal token new                  # one-time auth
modal deploy modal_app.py
# → prints  https://<workspace>--audio-trial-fastapi-app.modal.run
```

Frontend can either be:
- **Static-hosted** alongside Modal: build `pnpm build`, point your frontend hosting (Vercel / Netlify / Cloudflare Pages) at the Modal URL via `VITE_API_URL`.
- **Co-served** by the Modal app: the FastAPI app already mounts `frontend/dist` on `/` if it exists.

### Path C — AMD GPU cloud (when self-hosted models land in P1)

P0 MVP has **zero GPU workloads** — every adapter is either a cloud API or a small local-CPU model (Resemblyzer, pyannote/embedding-CPU). Deployment in P0 needs only a CPU server.

When P1 lands and self-hosted models start mattering (Parakeet ASR, pyannote diarization, Qwen-7B serving via vLLM), the candidate AMD GPU clouds are RunPod (MI300X serverless), TensorWave (bare-metal), Crusoe, Hot Aisle. See [`docs/PLAN.md`](docs/PLAN.md) §Backend hosting for the trade-offs vs Modal — the call there is "stay on Modal until traffic exceeds ~100 sustained RPS." This README will pick up the AMD wiring instructions when those P1 models actually exist.

### Exposing the URL — TL;DR

| You have | Easiest path |
|----------|--------------|
| A cloud VM with public IP | Path A + direct port (or Cloudflare for HTTPS) |
| A laptop / on-prem box behind NAT | Path A + Cloudflare Tunnel |
| No server at all | Path B (Modal) |
| GPU-heavy P1 workload (later) | Path C (RunPod / TensorWave) |

---

## Architecture (one screen)

```
React + Vite + TS  ─── HTTP REST ───▶  FastAPI on uvicorn
   sidebar:                              ├── /api/adapters
     Playground                          ├── /api/pipelines
     Pipelines                           ├── /api/clips
     Run                                 ├── /api/runs
     Corpus            ◀── WebSocket ─── └── /ws/run/{id}     (per-stage events)
     Settings                                    │
                                                 ▼
                                  ┌──────────────────────────┐
                                  │ adapters/registry.py     │
                                  │  · Deepgram (cloud)      │
                                  │  · Gladia (cloud)        │
                                  │  · Cartesia (cloud)      │
                                  │  · pyannote_verify (CPU) │
                                  │  · Resemblyzer (CPU)     │
                                  └──────────────────────────┘
```

Full data model and phasing in [`docs/PLAN.md`](docs/PLAN.md).

---

## Repo layout

```
backend/         FastAPI app, adapter registry, pipeline runner, clip storage
frontend/        Vite + React + TS — Playground / Pipelines / Run / Corpus / Settings
docs/            Plan, ADRs, deployment notes
modal_app.py     Optional Modal serverless entry (Path B)
docker-compose.yml + Dockerfile  (Path A)
```

Each new feature is its own commit on `main` (or a feature branch + PR).

---

## License

MIT — see `LICENSE`.
