# model-server

GPU-side companion to the trial-app. Hosts NeMo ASR models (Parakeet,
Canary-1B-flash, Canary-Qwen-2.5B) on the AMD-ROCm remote machine and
exposes them behind a single HTTP API.

The trial-app backend reaches this service via `MODEL_SERVER_URL` (set by
docker-compose to `http://model-server:9100` inside the stack).

## Routes

```
GET  /health                       liveness; reports ROCm presence
GET  /models                       per-model load status
POST /v1/transcribe?model=<id>     multipart audio → JSON
                                   {text, words, language, duration_s,
                                    model, latency_ms}
```

## Slice 1B status

| Model | Loader | Transcriber | Status |
|-------|--------|-------------|--------|
| `parakeet-tdt-1.1b` | ✓ | ✓ | working end-to-end |
| `canary-1b-flash` | stub | — | returns 503 until Slice 2 |
| `canary-qwen-2.5b` | stub | — | returns 503 until Slice 2 |

The Canary loaders are stubbed so the rest of the stack ships today.
Slice 2 fills them in once an AMD GPU is available for end-to-end
validation.

## Local dev (no GPU required)

```bash
cd model-server
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 9100
```

`/health` returns `rocm: false` on a CPU box; that's fine for development.
The trial-app's NeMo adapters work either way — you just won't see the
ROCm-accelerated speedups.

## In-container (production)

`Dockerfile` uses `rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_2.4.0`.
For non-MI300 GPUs (e.g. consumer 7900 XTX) set
`HSA_OVERRIDE_GFX_VERSION=10.3.0` (or the matching gfx code) at build time:

```bash
docker build --build-arg HSA_OVERRIDE_GFX_VERSION=10.3.0 -t model-server .
```

Models are pulled from Hugging Face on first request; persist the cache
via the `models` named volume in `docker-compose.yml`.
