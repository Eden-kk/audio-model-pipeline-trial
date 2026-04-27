"""AssemblyAI Universal-2 ASR adapter.

Cloud, async (upload then poll). Env: ASSEMBLYAI_API_KEY.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import httpx

_UPLOAD = "https://api.assemblyai.com/v2/upload"
_TRANSCRIPT = "https://api.assemblyai.com/v2/transcript"


class AssemblyAIAdapter:
    id = "assemblyai"
    category = "asr"
    display_name = "AssemblyAI Universal-2"
    hosting = "cloud"
    vendor = "AssemblyAI"

    inputs: List[Dict[str, str]] = [{"name": "audio", "type": "audio_file"}]
    outputs: List[Dict[str, str]] = [
        {"name": "text", "type": "text"},
        {"name": "words", "type": "word_timing"},
    ]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "language": {"type": "string", "default": "en"},
            "speaker_labels": {"type": "boolean", "default": True},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = None

    def _hdr(self) -> Dict[str, str]:
        k = os.environ.get("ASSEMBLYAI_API_KEY", "")
        if not k:
            raise RuntimeError("ASSEMBLYAI_API_KEY not set.")
        return {"authorization": k}

    async def transcribe(self, audio_path: str, config: dict) -> dict:
        language = config.get("language", "en")
        speaker_labels = bool(config.get("speaker_labels", True))

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(audio_path, "rb") as f:
                r1 = await client.post(_UPLOAD, headers=self._hdr(), content=f.read())
            r1.raise_for_status()
            audio_url = r1.json()["upload_url"]

            body = {"audio_url": audio_url, "speaker_labels": speaker_labels}
            if language and language != "auto":
                body["language_code"] = language
            r2 = await client.post(_TRANSCRIPT, headers={**self._hdr(),
                                   "content-type": "application/json"}, json=body)
            r2.raise_for_status()
            tid = r2.json()["id"]

            for _ in range(120):
                rp = await client.get(f"{_TRANSCRIPT}/{tid}", headers=self._hdr())
                rp.raise_for_status()
                bp = rp.json()
                if bp.get("status") == "completed":
                    body = bp
                    break
                if bp.get("status") == "error":
                    raise RuntimeError(f"AssemblyAI error: {bp.get('error')}")
                await asyncio.sleep(2.0)
            else:
                raise TimeoutError("AssemblyAI polling exceeded 240s")
        wall_s = time.perf_counter() - t0

        text = body.get("text", "") or ""
        words = []
        for w in body.get("words", []) or []:
            words.append({
                "word": w.get("text", ""),
                "start": float(w.get("start", 0)) / 1000.0,  # ms → s
                "end": float(w.get("end", 0)) / 1000.0,
                "confidence": w.get("confidence"),
                "speaker": w.get("speaker"),
            })
        duration_s = float(body.get("audio_duration", 0))
        cost_usd = round(duration_s / 60.0 * 0.0066, 6) if duration_s else None  # ~$0.40/hr

        return {
            "text": text,
            "words": words,
            "language": body.get("language_code", language),
            "duration_s": duration_s,
            "cost_usd": cost_usd,
            "wall_time_s": wall_s,
            "raw_response": body,
        }
