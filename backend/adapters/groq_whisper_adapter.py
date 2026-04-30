"""Groq Whisper-Large-v3-Turbo ASR adapter (LPU-accelerated).

Cloud, synchronous (single POST). Env: GROQ_API_KEY.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"


class GroqWhisperAdapter:
    id = "groq_whisper"
    category = "asr"
    display_name = "Groq Whisper (LPU)"
    hosting = "cloud"
    vendor = "Groq"

    inputs: List[Dict[str, str]] = [{"name": "audio", "type": "audio_file"}]
    outputs: List[Dict[str, str]] = [
        {"name": "text", "type": "text"},
        {"name": "words", "type": "word_timing"},
    ]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "model": {"type": "string", "default": "whisper-large-v3-turbo",
                      "enum": ["whisper-large-v3", "whisper-large-v3-turbo",
                               "distil-whisper-large-v3-en"]},
            "language": {"type": "string", "default": "auto"},
            "response_format": {"type": "string", "default": "verbose_json",
                                "enum": ["json", "verbose_json", "text"]},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = None
    is_streaming = True   # via chunked pseudo-stream (Groq has no native streaming endpoint)
    supported_languages: List[str] = ["auto", "en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "hi", "ar", "ru"]
    # pseudo-stream; language selected at call time only.
    multilang_realtime = False

    async def transcribe_stream(self, audio_path: str, config: dict):
        from ._pseudo_stream import pseudo_stream_chunks
        async for ev in pseudo_stream_chunks(self, audio_path, config):
            yield ev

    def _key(self) -> str:
        k = os.environ.get("GROQ_API_KEY", "")
        if not k:
            raise RuntimeError("GROQ_API_KEY not set.")
        return k

    async def transcribe(self, audio_path: str, config: dict) -> dict:
        model = config.get("model", "whisper-large-v3-turbo")
        language = config.get("language", "auto")
        response_format = config.get("response_format", "verbose_json")

        data = {"model": model, "response_format": response_format,
                "timestamp_granularities[]": "word"}
        if language and language != "auto":
            data["language"] = language

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(audio_path, "rb") as f:
                r = await client.post(
                    _ENDPOINT,
                    headers={"Authorization": f"Bearer {self._key()}"},
                    data=data,
                    files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                )
        r.raise_for_status()
        body = r.json()
        wall_s = time.perf_counter() - t0

        text = body.get("text", "")
        words = []
        for w in body.get("words", []) or []:
            words.append({
                "word": w.get("word", ""),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "confidence": None,
                "speaker": None,
            })
        duration_s = float(body.get("duration", 0))
        # Groq turbo: ~$0.04/hour ≈ $0.000667/min
        rate_per_min = 0.000667 if "turbo" in model else 0.00185
        cost_usd = round(duration_s / 60.0 * rate_per_min, 6) if duration_s else None

        return {
            "text": text,
            "words": words,
            "language": body.get("language", language),
            "duration_s": duration_s,
            "cost_usd": cost_usd,
            "wall_time_s": wall_s,
            "raw_response": body,
        }
