"""Speechmatics ASR adapter.

Cloud batch (job-based: submit, poll status, fetch transcript).
Env: SPEECHMATICS_API_KEY.
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

_BASE = "https://asr.api.speechmatics.com/v2/jobs/"


class SpeechmaticsAdapter:
    id = "speechmatics"
    category = "asr"
    display_name = "Speechmatics"
    hosting = "cloud"
    vendor = "Speechmatics"

    inputs: List[Dict[str, str]] = [{"name": "audio", "type": "audio_file"}]
    outputs: List[Dict[str, str]] = [
        {"name": "text", "type": "text"},
        {"name": "words", "type": "word_timing"},
    ]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "language": {"type": "string", "default": "en"},
            "diarize": {"type": "boolean", "default": True},
            "operating_point": {"type": "string", "default": "enhanced",
                                "enum": ["standard", "enhanced"]},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = None
    is_streaming = True   # via chunked pseudo-stream

    async def transcribe_stream(self, audio_path: str, config: dict):
        from ._pseudo_stream import pseudo_stream_chunks
        async for ev in pseudo_stream_chunks(self, audio_path, config):
            yield ev

    def _hdr(self) -> Dict[str, str]:
        k = os.environ.get("SPEECHMATICS_API_KEY", "")
        if not k:
            raise RuntimeError("SPEECHMATICS_API_KEY not set.")
        return {"Authorization": f"Bearer {k}"}

    async def transcribe(self, audio_path: str, config: dict) -> dict:
        language = config.get("language", "en")
        diarize = bool(config.get("diarize", True))
        operating_point = config.get("operating_point", "enhanced")

        cfg = {
            "type": "transcription",
            "transcription_config": {
                "language": language,
                "operating_point": operating_point,
                "diarization": "speaker" if diarize else "none",
            },
        }

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(audio_path, "rb") as f:
                r1 = await client.post(
                    _BASE, headers=self._hdr(),
                    files={
                        "data_file": (os.path.basename(audio_path), f, "audio/wav"),
                        "config": (None, _json.dumps(cfg), "application/json"),
                    },
                )
            r1.raise_for_status()
            jid = r1.json()["id"]

            for _ in range(120):
                rp = await client.get(f"{_BASE}{jid}", headers=self._hdr())
                rp.raise_for_status()
                status = rp.json().get("job", {}).get("status")
                if status == "done":
                    break
                if status == "rejected":
                    raise RuntimeError(f"Speechmatics rejected job: {rp.json()}")
                await asyncio.sleep(2.0)
            else:
                raise TimeoutError("Speechmatics polling exceeded 240s")

            r3 = await client.get(f"{_BASE}{jid}/transcript?format=json-v2",
                                  headers=self._hdr())
            r3.raise_for_status()
            body = r3.json()
        wall_s = time.perf_counter() - t0

        results = body.get("results", [])
        text_parts: List[str] = []
        words: List[Dict[str, Any]] = []
        for r in results:
            if r.get("type") != "word":
                continue
            alts = r.get("alternatives", [])
            if not alts:
                continue
            top = alts[0]
            tok = top.get("content", "")
            text_parts.append(tok)
            words.append({
                "word": tok,
                "start": float(r.get("start_time", 0.0)),
                "end": float(r.get("end_time", 0.0)),
                "confidence": top.get("confidence"),
                "speaker": top.get("speaker"),
            })
        text = " ".join(text_parts)
        duration_s = float(body.get("metadata", {}).get("duration", 0))
        # Standard ~$0.0083/min, enhanced ~$0.0167/min
        rate = 0.0167 if operating_point == "enhanced" else 0.0083
        cost_usd = round(duration_s / 60.0 * rate, 6) if duration_s else None

        return {
            "text": text,
            "words": words,
            "language": body.get("metadata", {}).get("transcription_config", {}).get("language", language),
            "duration_s": duration_s,
            "cost_usd": cost_usd,
            "wall_time_s": wall_s,
            "raw_response": body,
        }
