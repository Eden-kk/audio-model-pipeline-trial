# Deploying on an AMD-ROCm remote machine

End-to-end walkthrough for spinning up the trial app on an AMD GPU box and
exposing one HTTPS URL. This is the **canonical** deployment path; see the
README for the brief overview and the no-AMD fallback.

The stack:

```
[your laptop] ─ HTTPS ─▶ [Caddy:443] ─▶ [trial-app:8000] ──▶ [model-server:9100]
                                              │                ├ parakeet-tdt-1.1b
                                              │                ├ canary-1b-flash
                                              ▼                └ canary-qwen-2.5b
                                       SPA static (frontend/dist
                                        baked into trial-app image)
```

Three containers, one bridge network, one public port (443).

---

## 1. Hardware

Tested target: AMD Instinct MI300X. Will also work on:

| GPU | `HSA_OVERRIDE_GFX_VERSION` | VRAM | Notes |
|-----|----------------------------|------|-------|
| MI300X | (unset — auto-detected) | 192 GB | Best fit; runs all three models concurrently |
| MI250X | (unset) | 64 GB / GCD | Plenty of room |
| MI210 | (unset) | 64 GB | Fine |
| **Radeon 7900 XTX (Navi 31, gfx1100)** | `11.0.0` | 24 GB | Consumer card; one model at a time |
| **Radeon 7900 XT** | `11.0.0` | 20 GB | Consumer; tight on Canary-Qwen |
| Radeon 6800 XT (Navi 22, gfx1030) | `10.3.0` | 16 GB | Parakeet only; Canary-1B may OOM |

CPU-only fallback works for development (the model-server falls back
gracefully) — but transcription latency will be 5–20× slower.

---

## 2. Host prerequisites

```bash
# Verify ROCm visible to the kernel
sudo rocminfo | grep "Marketing Name"             # should print your GPU
ls -l /dev/kfd /dev/dri/renderD*                  # device files exist
sudo usermod -aG render,video $USER && newgrp video

# Docker + compose
sudo apt install -y docker.io docker-compose-plugin
docker --version && docker compose version
```

A first-time test that the container can see the GPU:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add=video \
    rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_2.4.0 \
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Should print `True 1` (or higher GPU count). If `False`, fix host
permissions before continuing.

---

## 3. Get the code + configure

```bash
git clone https://github.com/Eden-kk/audio-model-pipeline-trial.git
cd audio-model-pipeline-trial
cp .env.example .env
$EDITOR .env
```

Required keys in `.env`:

| Variable | Required | What it does |
|----------|----------|--------------|
| `PUBLIC_HOST` | yes | DNS hostname (e.g. `audio-trial.example.com`) — Caddy uses this for ACME |
| `PUBLIC_ORIGIN` | yes | `https://${PUBLIC_HOST}` — backend CORS allowlist |
| `LETSENCRYPT_EMAIL` | yes | For TLS cert renewal notices |
| `HSA_OVERRIDE_GFX_VERSION` | sometimes | Set per the hardware table above; leave blank for MI300X |
| `DEEPGRAM_API_KEY` *etc.* | optional | Each cloud adapter only activates when its key is present; the app shows a clear "no key" state otherwise |
| `HF_TOKEN` | recommended | Needed for `pyannote_verify` (accept the gated license at https://huggingface.co/pyannote/embedding) |

---

## 4. DNS + first launch

Point `PUBLIC_HOST`'s A-record at the server's public IP **before**
launching — Caddy needs to complete the ACME HTTP challenge on port 80.
If you can't open public ports, see §6 (Cloudflare Tunnel).

```bash
docker compose up -d --build
docker compose ps                   # all 3 services should be Up
docker compose logs -f caddy        # watch Let's Encrypt issue the cert
```

When Caddy says `certificate obtained`, your URL is live:

```bash
curl https://audio-trial.example.com/api/health
# {"status":"ok","version":"0.1.0"}
```

Open `https://audio-trial.example.com` in a browser — you should land on
the Playground page with the model dropdown populated.

---

## 5. First transcription

1. Open the deployed URL.
2. Click **Playground** in the sidebar.
3. Pick `parakeet` (self-host) or `deepgram` (cloud — only if you set the
   key) from the adapter dropdown.
4. Click record, say a sentence, click stop.
5. Click **Run**. You should see the transcript + per-stage latency badge.

If the first Parakeet call takes ~90 s, that's the model loading from
HuggingFace into the `models` volume. Subsequent calls are warm
(<1 s on MI300X for short clips).

---

## 6. URL exposure variants

| Your situation | Path |
|----------------|------|
| Public IP + ports 80/443 open | Use the docker-compose stack as-is. Caddy gets ACME. |
| Public IP, ports closed | Skip Caddy, run only `trial-app` + `model-server`; put a Cloudflare Tunnel in front: `cloudflared tunnel --url http://localhost:8000`. |
| No public IP at all (dorm / on-prem) | Same — Cloudflare Tunnel solves NAT. |
| Internal use only | SSH tunnel: `ssh -L 8000:localhost:8000 user@server`, visit `http://localhost:8000` locally. Skip Caddy entirely. |

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|--------|-------------|-----|
| `model-server` exits with `RuntimeError: HIP error: invalid device function` | `HSA_OVERRIDE_GFX_VERSION` not set for your card | Set it in `.env` per §1, then `docker compose up -d --build model-server` |
| `model-server unreachable at http://model-server:9100` (visible in trial-app stage panel) | model-server still loading, or container crashed | `docker compose logs model-server` |
| Caddy fails to obtain cert | DNS not yet propagated, or port 80 blocked | `dig $PUBLIC_HOST` to confirm A-record; check firewall |
| `403` when loading the SPA | Frontend dist not baked into image | `docker compose build trial-app` then `up -d` |
| `500` on `/api/runs` for a cloud adapter | API key missing | check `.env` keys; restart trial-app |
| Canary models return 503 "model not yet implemented" | Slice 1B intentionally stubbed Canary loaders | Slice 2 will fill these in once validated on a real AMD GPU |
| Out-of-memory loading model | GPU too small for that model | Pick a different model or upgrade the GPU |

---

## 8. Operating notes

- **Persistence.** `models` volume keeps HuggingFace + NeMo caches across image rebuilds. `trial-data` keeps clip uploads and run JSONL logs. Don't `docker compose down -v` unless you want to nuke them.
- **Updates.** `git pull && docker compose up -d --build` rebuilds only changed services.
- **Logs.** `docker compose logs -f --tail=100`. Caddy logs are JSON; trial-app and model-server are uvicorn logs.
- **Cost.** All cloud adapters are pay-per-use; the trial app prints a per-call cost estimate. The model-server itself just costs your GPU's wall-clock time.

---

## 9. What's next

When Slice 2 lands you'll get:
- Canary-1B-flash and Canary-Qwen-2.5B loaders validated on real AMD hardware
- Pre-built recipes (ASR-only, TTS-only, Speaker-verify-only, ASR+TTS roundtrip)

Until then, the Playground page is the main way in — it works against any of the 12 registered adapters individually.
