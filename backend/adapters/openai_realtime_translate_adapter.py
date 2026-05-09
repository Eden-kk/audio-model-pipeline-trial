"""OpenAI Realtime Translate adapter — per-utterance speech-to-speech translation.

Per-utterance WSS: one connection opened per flush frame, closed after response.done.

Env: OPENAI_API_KEY
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np
from scipy.signal import resample_poly  # type: ignore[import]
import websockets  # type: ignore[import]

log = logging.getLogger("trial-app.openai_realtime_translate")

_WS_URL = "wss://api.openai.com/v1/realtime?model={model}"
_CHUNK = 4096


class OpenAIRealtimeTranslateAdapter:
    id = "openai_realtime_translate"
    category = "realtime_omni"
    display_name = "OpenAI Realtime Translate"
    hosting = "cloud"
    vendor = "OpenAI"
    is_streaming = True
    cost_per_call_estimate_usd: Optional[float] = 0.034

    inputs: List[Dict[str, str]] = [{"name": "media", "type": "media_stream"}]
    outputs: List[Dict[str, str]] = [{"name": "events", "type": "omni_event"}]

    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "source_language": {
                "type": "string",
                "default": "auto",
                "description": "BCP-47 or 'auto'",
            },
            "target_language": {
                "type": "string",
                "default": "en",
                "enum": ["en", "de", "fr", "es", "nl", "hi", "ja", "ko", "pt", "it", "pl", "ru", "zh"],
            },
            "voice": {
                "type": "string",
                "default": "alloy",
                "enum": ["alloy", "echo", "shimmer", "fable", "onyx", "nova"],
            },
            "model": {
                "type": "string",
                "default": "gpt-realtime",
                "description": (
                    "Carrier realtime model used with a translation system prompt. "
                    "gpt-realtime-translate is broken server-side as of 2026-05-08."
                ),
            },
        },
    }

    def __init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            log.warning(
                "OPENAI_API_KEY not set — openai_realtime_translate adapter will fail on first call"
            )

    def _api_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return key

    async def omni_session(
        self,
        media_iter: AsyncIterator[Dict[str, Any]],
        *,
        config: dict,
        context_iter: Optional[AsyncIterator[Any]] = None,  # not used; translation is stateless
        abort_event: Optional[Any] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        source_language = config.get("source_language", "auto")
        target_language = config.get("target_language", "en")
        voice = config.get("voice", "alloy")
        model = config.get("model", "gpt-realtime")
        api_key = self._api_key()

        pcm_buf = bytearray()

        async for frame in media_iter:
            if frame.get("type") == "audio":
                payload = frame.get("payload")
                if payload:
                    pcm_buf.extend(payload)
                continue

            if frame.get("type") != "flush":
                continue

            if abort_event is not None and abort_event.is_set():
                yield {"type": "done", "latency_ms": 0.0, "cost_usd": 0.0, "aborted": True}
                pcm_buf = bytearray()
                continue

            t0 = time.perf_counter()

            pcm_16k = np.frombuffer(bytes(pcm_buf), dtype="int16")
            pcm_24k = resample_poly(pcm_16k, 3, 2).astype("int16")
            pcm_bytes = pcm_24k.tobytes()

            session: Dict[str, Any] = {
                "modalities": ["audio", "text"],
                "instructions": (
                    f"Translate the speaker's speech into {target_language}. "
                    "Output only the translation, no additional commentary."
                ),
                "voice": voice,
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": None,
                **({"language": source_language} if source_language != "auto" else {}),
            }

            async with websockets.connect(
                _WS_URL.format(model=model),
                additional_headers={
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
                ping_interval=20,
            ) as ws:
                await ws.send(json.dumps({"type": "session.update", "session": session}))

                for i in range(0, len(pcm_bytes), _CHUNK):
                    chunk = pcm_bytes[i: i + _CHUNK]
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }))

                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await ws.send(json.dumps({"type": "response.create"}))

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    t = msg.get("type", "")
                    if t == "response.audio.delta":
                        yield {"type": "audio_b64", "data": msg["delta"], "sample_rate": 24000}
                    elif t == "response.audio_transcript.delta":
                        yield {"type": "transcript", "text": msg.get("delta", ""), "is_final": False}
                    elif t == "response.done":
                        elapsed = time.perf_counter() - t0
                        yield {
                            "type": "done",
                            "latency_ms": elapsed * 1000,
                            "cost_usd": elapsed / 60.0 * (self.cost_per_call_estimate_usd or 0.034),
                        }
                        break

            pcm_buf = bytearray()
