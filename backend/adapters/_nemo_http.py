"""Shared HTTP client for self-hosted NeMo ASR models.

The audio-trial backend talks to a separate `model-server` that hosts
Parakeet + Canary models behind a unified HTTP API. Two deploy targets
ship today:

  - Modal       (CUDA L4, default — `modal deploy model-server/modal_app.py`)
  - AMD-ROCm    (`docker compose -f model-server/docker-compose.yml up -d`)

Both speak the same wire shape; only the URL changes. Adapters pick
which target to hit via env (resolved by ``model_server_url()`` below):

  MODEL_SERVER_BACKEND=modal|amd   declares which target to use.
                                   Defaults to "modal".
  MODEL_SERVER_MODAL_URL=https://… public Modal URL printed by deploy.
  MODEL_SERVER_AMD_URL=http://…    AMD docker-compose host:port (default
                                   http://localhost:9100).
  MODEL_SERVER_URL=…               legacy single-knob; if set, beats both
                                   of the above (handy in CI / one-offs).

Wire contract (model-server/server.py for AMD, modal_app.py for Modal):
  POST {url}/v1/transcribe?model=<id>
       multipart/form-data with `file` = audio bytes
  →   {text, words, language, duration_s, model, latency_ms}
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx


def model_server_url() -> str:
    """Resolve the model-server URL, honouring the legacy single-knob first."""
    legacy = os.environ.get("MODEL_SERVER_URL")
    if legacy:
        return legacy.rstrip("/")
    backend = os.environ.get("MODEL_SERVER_BACKEND", "modal").strip().lower()
    if backend == "amd":
        return os.environ.get("MODEL_SERVER_AMD_URL",
                              "http://localhost:9100").rstrip("/")
    if backend == "modal":
        # Modal default of localhost:9100 is intentional — you HAVE to set
        # MODEL_SERVER_MODAL_URL after `modal deploy` prints the public URL.
        # The localhost fallback just lets a dev poke the API w/ a fake server.
        return os.environ.get("MODEL_SERVER_MODAL_URL",
                              "http://localhost:9100").rstrip("/")
    raise RuntimeError(
        f"MODEL_SERVER_BACKEND={backend!r} not understood. "
        f"Set to 'modal' or 'amd', or set MODEL_SERVER_URL directly."
    )


async def transcribe_via_model_server(
    audio_path: str,
    *,
    model: str,
    timeout_s: float = 120.0,
) -> dict:
    """Forward an audio file to the model-server and normalise the response.

    Raises a clear error if the model-server is unreachable so the trial-app
    UI can render "model-server offline" rather than a generic timeout.
    """
    url = f"{model_server_url()}/v1/transcribe"
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            with open(audio_path, "rb") as f:
                resp = await client.post(
                    url,
                    params={"model": model},
                    files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                )
    except httpx.ConnectError as e:
        backend = os.environ.get("MODEL_SERVER_BACKEND", "modal")
        raise RuntimeError(
            f"model-server unreachable at {url}: {e}. "
            f"Backend is MODEL_SERVER_BACKEND={backend!r}. Try:\n"
            f"  • Modal:  `modal deploy model-server/modal_app.py` and set "
            f"MODEL_SERVER_MODAL_URL to the printed URL\n"
            f"  • AMD:    `docker compose -f model-server/docker-compose.yml "
            f"up -d` (then MODEL_SERVER_AMD_URL=http://localhost:9100)"
        ) from e
    if resp.status_code == 503:
        raise RuntimeError(
            f"model-server reports model '{model}' is not loaded. "
            f"Response: {resp.text[:200]}"
        )
    resp.raise_for_status()
    body = resp.json()
    wall_s = time.perf_counter() - t0

    return {
        "text": body.get("text", ""),
        "words": body.get("words", []) or [],
        "language": body.get("language", "en"),
        "duration_s": float(body.get("duration_s", 0.0)),
        "cost_usd": 0.0,  # self-host
        "wall_time_s": wall_s,
        "model_server_latency_ms": body.get("latency_ms"),
        "raw_response": body,
    }


# Common port declarations (most NeMo ASR adapters share these)
NEMO_INPUTS: List[Dict[str, str]] = [{"name": "audio", "type": "audio_file"}]
NEMO_OUTPUTS: List[Dict[str, str]] = [
    {"name": "text", "type": "text"},
    {"name": "words", "type": "word_timing"},
]


def nemo_config_schema(default_lang: str = "en") -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "language": {"type": "string", "default": default_lang,
                         "description": "BCP-47 code or 'auto' (model-dependent)."},
        },
    }
