"""Gladia ASR adapter.

Cloud, async (upload then poll). Env: GLADIA_API_KEY.
Wire shape adapted from ambient-deploy/benchmarks/adapters/gladia.py.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import httpx

_BASE = "https://api.gladia.io"


class GladiaAdapter:
    id = "gladia"
    category = "asr"
    display_name = "Gladia"
    hosting = "cloud"
    vendor = "Gladia"

    inputs: List[Dict[str, str]] = [{"name": "audio", "type": "audio_file"}]
    outputs: List[Dict[str, str]] = [
        {"name": "text", "type": "text"},
        {"name": "words", "type": "word_timing"},
    ]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "language": {"type": "string", "default": "auto",
                         "description": "BCP-47 code or 'auto'."},
            "diarize": {"type": "boolean", "default": True},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = None  # Gladia ~$0.005/min
    is_streaming = True   # via chunked pseudo-stream
    supported_languages: List[str] = ["auto", "en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "hi", "ar", "ru"]
    # gladia auto mode detects language switches within a single session.
    multilang_realtime = True

    async def transcribe_stream(self, audio_path: str, config: dict):
        from ._pseudo_stream import pseudo_stream_chunks
        async for ev in pseudo_stream_chunks(self, audio_path, config):
            yield ev

    def _key(self) -> str:
        k = os.environ.get("GLADIA_API_KEY", "")
        if not k:
            raise RuntimeError("GLADIA_API_KEY not set.")
        return k

    async def _upload(self, client: httpx.AsyncClient, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            r = await client.post(
                f"{_BASE}/v2/upload",
                headers={"x-gladia-key": self._key()},
                files={"audio": (os.path.basename(audio_path), f, "audio/wav")},
            )
        r.raise_for_status()
        return r.json()["audio_url"]

    async def _start_job(self, client: httpx.AsyncClient,
                         audio_url: str, *, language: str, diarize: bool) -> str:
        body = {"audio_url": audio_url, "diarization": diarize}
        if language and language != "auto":
            body["language_config"] = {"languages": [language]}
        r = await client.post(
            f"{_BASE}/v2/pre-recorded",
            headers={"x-gladia-key": self._key(), "Content-Type": "application/json"},
            json=body,
        )
        r.raise_for_status()
        return r.json()["id"]

    async def _poll(self, client: httpx.AsyncClient, job_id: str) -> dict:
        # Poll up to ~120s
        for _ in range(60):
            r = await client.get(f"{_BASE}/v2/pre-recorded/{job_id}",
                                 headers={"x-gladia-key": self._key()})
            r.raise_for_status()
            body = r.json()
            if body.get("status") == "done":
                return body
            if body.get("status") == "error":
                raise RuntimeError(f"Gladia error: {body}")
            await asyncio.sleep(2.0)
        raise TimeoutError("Gladia polling exceeded 120s")

    async def transcribe(self, audio_path: str, config: dict) -> dict:
        language = config.get("language", "auto")
        diarize = bool(config.get("diarize", True))

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            url = await self._upload(client, audio_path)
            job_id = await self._start_job(client, url, language=language, diarize=diarize)
            body = await self._poll(client, job_id)
        wall_s = time.perf_counter() - t0

        result = body.get("result", {})
        transcription = result.get("transcription", {})
        text = transcription.get("full_transcript", "")
        utterances = transcription.get("utterances", []) or []
        words: List[Dict[str, Any]] = []
        for u in utterances:
            for w in u.get("words", []) or []:
                words.append({
                    "word": w.get("word", ""),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "confidence": w.get("confidence"),
                    "speaker": str(u["speaker"]) if "speaker" in u else None,
                })
        meta = result.get("metadata", {})
        duration_s = float(meta.get("audio_duration", 0.0))
        cost_usd = round(duration_s / 60.0 * 0.005, 6) if duration_s else None
        detected_lang = transcription.get("languages", [language])[0] if transcription.get("languages") else language

        return {
            "text": text,
            "words": words,
            "language": detected_lang,
            "duration_s": duration_s,
            "cost_usd": cost_usd,
            "wall_time_s": wall_s,
            "raw_response": body,
        }
