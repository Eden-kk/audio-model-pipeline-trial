"""Cartesia Sonic-3 TTS adapter.

Streaming HTTP. Returns raw 16-bit-LE PCM bytes at 16 kHz mono by default.
Measures both first-byte latency (TTFA) and full-render latency.

Env: CARTESIA_API_KEY.
"""
from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional

import httpx

_ENDPOINT = "https://api.cartesia.ai/tts/bytes"
_DEFAULT_VOICE = "a0e99841-438c-4a64-b679-ae501e7d6091"  # "Barbershop Man"


class CartesiaTTSAdapter:
    id = "cartesia_tts"
    category = "tts"
    display_name = "Cartesia Sonic-3"
    hosting = "cloud"
    vendor = "Cartesia"

    inputs: List[Dict[str, str]] = [{"name": "text", "type": "text"}]
    outputs: List[Dict[str, str]] = [{"name": "audio", "type": "audio_stream"}]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "default": "sonic-3"},
            "voice_id": {"type": "string", "default": _DEFAULT_VOICE},
            "language": {"type": "string", "default": "en"},
            "sample_rate": {"type": "integer", "default": 16000,
                            "enum": [16000, 22050, 24000, 44100]},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = None  # Cartesia ~$1/1M chars

    def _key(self) -> str:
        k = os.environ.get("CARTESIA_API_KEY", "")
        if not k:
            raise RuntimeError("CARTESIA_API_KEY not set.")
        return k

    async def synthesize(self, text: str, config: dict) -> dict:
        model_id = config.get("model_id", "sonic-3")
        voice_id = config.get("voice_id", _DEFAULT_VOICE)
        language = config.get("language", "en")
        sample_rate = int(config.get("sample_rate", 16000))

        payload = {
            "model_id": model_id,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
            },
            "language": language,
        }
        headers = {
            "X-API-Key": self._key(),
            "Cartesia-Version": "2024-11-13",
            "Content-Type": "application/json",
        }

        chunks: List[bytes] = []
        first_byte_ms: Optional[float] = None
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", _ENDPOINT, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        if first_byte_ms is None:
                            first_byte_ms = (time.perf_counter() - t0) * 1000
                        chunks.append(chunk)
        full_render_ms = (time.perf_counter() - t0) * 1000

        audio_bytes = b"".join(chunks)
        duration_s = (len(audio_bytes) / 2) / sample_rate  # 16-bit mono
        cost_usd = round(len(text) / 1_000_000.0 * 1.00, 6)  # ~$1 / 1M chars

        return {
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            "mime": "audio/L16",  # raw PCM 16-bit
            "sample_rate": sample_rate,
            "duration_s": duration_s,
            "first_byte_ms": first_byte_ms,
            "full_render_ms": full_render_ms,
            "cost_usd": cost_usd,
        }
