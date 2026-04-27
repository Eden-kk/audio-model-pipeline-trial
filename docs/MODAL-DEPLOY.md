# Local trial-app + Modal model-server

For testing without an AMD-ROCm machine. The trial-app (FastAPI + SPA)
runs on your laptop; the heavy NeMo models run on Modal's NVIDIA L4 GPU
behind the same `/v1/transcribe?model=<id>` API the AMD model-server
exposes. **No code changes required** — only one env var.

```
[your laptop:5173]  ─ HTTP ─▶  [your laptop:8000]  ─ HTTPS ─▶  [Modal L4]
       SPA (pnpm dev)              FastAPI                       NeMo:
                                   trial-app                       parakeet-tdt-1.1b
                                                                   canary-1b-flash
                                                                   canary-qwen-2.5b
```

---

## 1. Deploy the model-server to Modal (one-time)

```bash
pip install modal
modal token new                              # opens browser; one-time auth
cd model-server
modal deploy modal_app.py
```

Output ends with the public URL — copy it:

```
✓ App deployed. Web endpoint:
  https://<your-workspace>--audio-trial-model-server-modelserver-fastapi.modal.run
```

Sanity check:

```bash
curl https://<your-workspace>--audio-trial-model-server-modelserver-fastapi.modal.run/health
# {"status":"ok","version":"0.1.0","rocm":false}
```

`rocm:false` is correct — Modal runs CUDA, not ROCm. NeMo loads the same
way; only the inference backend differs. The user-facing API is identical.

> **Cost note.** Modal L4 is ~$0.80/hr while the container is up. With
> `scaledown_window=600`, a cold container shuts down 10 min after the
> last request. Casual testing is well under $1/day.

---

## 2. Run the trial-app locally

In one terminal, start the backend:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
$EDITOR .env       # fill in any cloud API keys you have

# Point the NeMo adapters at Modal:
export MODEL_SERVER_URL=https://<your-workspace>--audio-trial-model-server-modelserver-fastapi.modal.run

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

In another terminal, start the frontend:

```bash
cd frontend
pnpm install      # first time only
pnpm dev          # http://localhost:5173
```

Open `http://localhost:5173`. Sidebar → **Playground**:

1. Pick `parakeet` from the adapter dropdown.
2. Click record, say a sentence, stop.
3. Click **Run**. First run cold-starts Modal (~60–90 s including model download); subsequent runs are warm (~1 s for short clips).

The Modal deploy log will show the request:

```bash
modal app logs audio-trial-model-server --tail 30
```

---

## 3. Putting `MODEL_SERVER_URL` in `.env`

Avoid retyping each session — add it to `backend/.env`:

```env
MODEL_SERVER_URL=https://<your-workspace>--audio-trial-model-server-modelserver-fastapi.modal.run
```

The backend reads it via `python-dotenv` at startup.

---

## 4. What works on Modal vs AMD

| Model | AMD (ROCm) | Modal (CUDA) |
|-------|-----------|--------------|
| `parakeet-tdt-1.1b` | works (Slice 1B) | works |
| `canary-1b-flash` | stubbed → 503 (Slice 2) | stubbed → 503 (Slice 2) |
| `canary-qwen-2.5b` | stubbed → 503 (Slice 2) | stubbed → 503 (Slice 2) |

Adding the Canary loaders is the same code on both backends — Slice 2.
For now Parakeet alone gives you a working self-host ASR row in the
Playground.

---

## 5. Troubleshooting

| Symptom | Fix |
|--------|-----|
| `model-server unreachable at https://...modal.run` | Run `modal app logs audio-trial-model-server`; if the app isn't deployed run `modal deploy modal_app.py` again |
| Cold-start takes >2 min | Parakeet pulls ~2 GB from HuggingFace on first call; second call is warm. Preload by hitting `/health` then `POST /v1/transcribe` with a tiny clip once after each deploy |
| `503 model 'canary-...' not yet implemented` | Expected; Slice 2 will land the loaders |
| Modal billing | Set a spend cap in the Modal dashboard. With `scaledown_window=600` and casual use this should stay sub-$1/day |

---

## 6. Switching back to AMD later

When the AMD remote machine comes online:

1. Set `MODEL_SERVER_URL=http://model-server:9100` in `.env` (or the LAN IP).
2. Use the docker-compose stack from [`AMD-DEPLOY.md`](AMD-DEPLOY.md).

The trial-app's adapters don't care which backend is on the other end of
`MODEL_SERVER_URL`; the wire shape is the same.
