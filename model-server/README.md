# model-server

GPU-side companion to the trial-app. Hosts NeMo ASR models (Parakeet,
Canary-1B-flash, Canary-Qwen-2.5B) and exposes them behind a single
HTTP API.

Two deploy targets ship — pick based on what hardware you have:

| Target           | When to use                                         | One-liner                                                  |
| ---------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| **Modal** (default) | No local GPU; use NVIDIA L4 in the cloud (~$1/h) | `modal deploy modal_app.py`                                |
| **AMD ROCm**     | Local AMD GPU box (MI300X / MI250 / 7900 XTX)        | `docker compose -f docker-compose.yml up -d`               |

The trial-app's NeMo adapters pick which target to call via env (set on
the *trial-app* host, not here):

```
MODEL_SERVER_BACKEND=modal|amd          # which target, default modal
MODEL_SERVER_MODAL_URL=https://...      # public URL printed by `modal deploy`
MODEL_SERVER_AMD_URL=http://...:9100    # AMD docker-compose host:port
```

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

## AMD ROCm deploy (docker compose)

`Dockerfile` uses `rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_2.4.0`.
The provided `docker-compose.yml` exposes `/dev/kfd` + `/dev/dri`, joins
the `video` and `render` groups, and sets `shm_size: 8g` (NeMo's data
loader needs it).

```bash
# 1. (optional) drop your HF token in a .env file next to this README:
echo "HF_TOKEN=hf_..." > .env

# 2. for non-MI300 cards, also set HSA_OVERRIDE_GFX_VERSION:
echo "HSA_OVERRIDE_GFX_VERSION=11.0.0" >> .env    # 7900 XTX example

# 3. build + start
docker compose up -d

# 4. verify
curl http://localhost:9100/health        # → {"status":"ok","rocm":true,...}
curl -F file=@some.wav http://localhost:9100/v1/transcribe?model=parakeet-tdt-1.1b
```

Then tell the trial-app to use it:

```bash
export MODEL_SERVER_BACKEND=amd
export MODEL_SERVER_AMD_URL=http://<host-ip>:9100
```

Models pull from Hugging Face on first request; the `audio-trial-model-cache`
named volume keeps them across container rebuilds.

### CPU-only fallback

Comment out the `devices:` + `group_add:` blocks in `docker-compose.yml`.
NeMo / pyannote / Resemblyzer all degrade gracefully to CPU; speed drops
~10×, accuracy unchanged. Useful for plumbing tests on a laptop.

## Modal deploy (default)

```bash
modal deploy modal_app.py
# → prints https://<workspace>--audio-trial-model-server-fastapi.modal.run
```

Then on the trial-app host:

```bash
export MODEL_SERVER_BACKEND=modal     # or omit; modal is the default
export MODEL_SERVER_MODAL_URL=https://...   # the printed URL
```
