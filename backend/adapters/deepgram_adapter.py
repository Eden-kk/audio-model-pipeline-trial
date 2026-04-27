"""Deepgram Nova-3 ASR adapter for the audio-trial backend.

Wire-protocol rewritten from ambient-deploy/benchmarks/adapters/deepgram.py but
conforms to the local Adapter protocol (category, typed ports, async transcribe).

Env: DEEPGRAM_API_KEY
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

_ENDPOINT = "https://api.deepgram.com/v1/listen"

_DEFAULT_PARAMS = {
    "model": "nova-3",
    "smart_format": "true",
    "punctuate": "true",
    "diarize": "true",
    "utterances": "true",
}


class DeepgramAdapter:
    # ── Adapter identity ────────────────────────────────────────────────
    id = "deepgram"
    category = "asr"
    display_name = "Deepgram Nova-3"
    hosting = "cloud"
    vendor = "Deepgram"

    inputs: List[Dict[str, str]] = [
        {"name": "audio", "type": "audio_file"},
    ]
    outputs: List[Dict[str, str]] = [
        {"name": "text", "type": "text"},
        {"name": "words", "type": "word_timing"},
    ]

    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "default": "en",
                "description": "BCP-47 language code or 'auto' for auto-detect.",
            },
            "model": {
                "type": "string",
                "default": "nova-3",
                "description": "Deepgram model variant.",
            },
            "diarize": {
                "type": "boolean",
                "default": True,
                "description": "Enable speaker diarization.",
            },
        },
    }
    cost_per_call_estimate_usd: Optional[float] = None  # computed per call from duration

    # ── Construction ────────────────────────────────────────────────────
    def __init__(self) -> None:
        # Key is read lazily at call time so the adapter can be registered at
        # import time even when the env var is absent (e.g. during tests).
        pass

    def _api_key(self) -> str:
        key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is not set. "
                "Add it to backend/.env before calling Deepgram."
            )
        return key

    # ── Core HTTP call ──────────────────────────────────────────────────
    async def _call(
        self,
        audio_path: str,
        *,
        language: str,
        model: str,
        diarize: bool,
    ) -> dict:
        lang_param = language if language != "auto" else "multi"
        params = {
            **_DEFAULT_PARAMS,
            "model": model,
            "language": lang_param,
            "diarize": "true" if diarize else "false",
            "detect_language": "true" if lang_param == "multi" else "false",
        }
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                _ENDPOINT,
                params=params,
                headers={
                    "Authorization": f"Token {self._api_key()}",
                    "Content-Type": "audio/wav",
                },
                content=audio_bytes,
            )
        resp.raise_for_status()
        return resp.json()

    # ── Public transcribe ───────────────────────────────────────────────
    async def transcribe(self, audio_path: str, config: dict) -> dict:
        """Run ASR via Deepgram Nova-3 and return a normalised result dict.

        Returns:
            {text, words, language, duration_s, cost_usd, raw_response}
        """
        language = config.get("language", "en")
        model = config.get("model", "nova-3")
        diarize = bool(config.get("diarize", True))

        t0 = time.perf_counter()
        body = await self._call(
            audio_path, language=language, model=model, diarize=diarize
        )
        wall_s = time.perf_counter() - t0

        results = body.get("results", {})
        channels = results.get("channels", [])
        if not channels:
            raise ValueError("Deepgram returned no channels in response")

        alt = channels[0]["alternatives"][0]
        text = alt.get("transcript", "")
        detected_language = channels[0].get("detected_language", language)

        words = []
        for w in alt.get("words", []) or []:
            words.append({
                "word": w.get("word", ""),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "confidence": w.get("confidence"),
                "speaker": str(w["speaker"]) if "speaker" in w else None,
            })

        duration_s = float(
            results.get("metadata", {}).get("duration")
            or body.get("metadata", {}).get("duration", 0)
        )
        # Nova-3 batch pricing: ~$0.0043 / minute
        cost_usd = round(duration_s / 60.0 * 0.0043, 6) if duration_s else None

        return {
            "text": text,
            "words": words,
            "language": detected_language,
            "duration_s": duration_s,
            "cost_usd": cost_usd,
            "wall_time_s": wall_s,
            "raw_response": body,
        }
