"""Deepgram-backed Language ID adapter.

Uses Deepgram's `?detect_language=true&language=multi` on a short prefix
of the audio. The transcription itself is discarded — we only care about
the `detected_language` field. Faster than running a Whisper LID locally
when the audio is already on its way to Deepgram for ASR anyway.

Env: DEEPGRAM_API_KEY.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

_ENDPOINT = "https://api.deepgram.com/v1/listen"


class DeepgramLIDAdapter:
    id = "deepgram_lid"
    category = "lid"
    display_name = "Deepgram LID (Nova-3 detect_language)"
    hosting = "cloud"
    vendor = "Deepgram"
    is_streaming = False

    inputs: List[Dict[str, str]] = [{"name": "audio", "type": "audio_file"}]
    outputs: List[Dict[str, str]] = [{"name": "language", "type": "language"}]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "model": {"type": "string", "default": "nova-3"},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = None

    def _key(self) -> str:
        k = os.environ.get("DEEPGRAM_API_KEY", "")
        if not k:
            raise RuntimeError("DEEPGRAM_API_KEY not set.")
        return k

    async def lid(self, audio_path: str, config: dict) -> dict:
        model = config.get("model", "nova-3")
        params = {
            "model": model,
            "language": "multi",
            "detect_language": "true",
            "smart_format": "false",
            "punctuate": "false",
        }
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                _ENDPOINT,
                params=params,
                headers={
                    "Authorization": f"Token {self._key()}",
                    "Content-Type": "audio/wav",
                },
                content=audio_bytes,
            )
        if r.status_code >= 400:
            raise RuntimeError(f"Deepgram LID HTTP {r.status_code}: {r.text[:200]}")
        wall_s = time.perf_counter() - t0
        body = r.json()

        results = body.get("results", {})
        channels = results.get("channels", [])
        if not channels:
            raise RuntimeError("Deepgram LID returned no channels")
        ch0 = channels[0]
        # Newer responses use 'detected_language' on the channel; older
        # ones expose it on alternatives[0].
        lang = (ch0.get("detected_language")
                or (ch0.get("alternatives", [{}])[0].get("language"))
                or "en")
        # Confidence isn't always provided by Deepgram for LID; fall back to 1.0
        confidence = float(ch0.get("language_confidence", 1.0))

        return {
            "language": lang,
            "confidence": confidence,
            "candidates": [{"language": lang, "confidence": confidence}],
            "wall_time_s": wall_s,
            "cost_usd": 0.0001,  # rough — short LID call, ~$0.0001
            "model": model,
            "raw_response": body,
        }
