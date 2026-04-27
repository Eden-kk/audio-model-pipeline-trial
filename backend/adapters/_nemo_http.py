"""Shared HTTP client for self-hosted NeMo ASR models.

The audio-trial backend talks to a separate `model-server` container (Slice 1B)
that runs on the AMD-ROCm remote machine and hosts Parakeet, Canary-1B-flash,
Canary-Qwen-2.5B (and any other NeMo models added later) behind a unified
HTTP API.  This helper handles the wire shape; the per-model adapters just
declare their identity and call _transcribe_via_model_server(model="...").

Env: MODEL_SERVER_URL (default http://localhost:9100).

Model-server contract (matches Slice 1B's model-server/server.py):
  POST {MODEL_SERVER_URL}/v1/transcribe?model=<id>
       multipart/form-data with `file` = audio bytes
  →   {text, words, language, duration_s, model, latency_ms}
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx


def model_server_url() -> str:
    return os.environ.get("MODEL_SERVER_URL", "http://localhost:9100").rstrip("/")


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
        raise RuntimeError(
            f"model-server unreachable at {url}: {e}. "
            "Make sure the model-server container is up "
            "(docker compose ps model-server) and MODEL_SERVER_URL is correct."
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
