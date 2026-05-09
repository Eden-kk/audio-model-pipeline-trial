"""OpenAI Realtime Translate adapter — speech-to-speech translation.

Persistent WSS session with server_vad auto-segmenting: as the user
speaks, audio is forwarded continuously; the API detects end-of-utterance
at silence boundaries, auto-commits, and streams a translation response
back. Multi-utterance — the session stays open across pauses.

Uses the GA Realtime API (no `OpenAI-Beta` header). The carrier model is
`gpt-realtime` with a translation system prompt — `gpt-realtime-translate`
is broken server-side as of 2026-05-08 (inference_not_found_error).

Env: OPENAI_API_KEY
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
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
        context_iter: Optional[AsyncIterator[Any]] = None,
        abort_event: Optional[Any] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        target_language = config.get("target_language", "en")
        voice = config.get("voice", "alloy")
        model = config.get("model", "gpt-realtime")
        api_key = self._api_key()

        instructions = (
            f"Translate the speaker's speech into {target_language}. "
            "Output only the translation, no additional commentary."
        )

        async with websockets.connect(
            _WS_URL.format(model=model),
            additional_headers={"Authorization": f"Bearer {api_key}"},
            ping_interval=20,
        ) as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": instructions,
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": "whisper-1"},
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": 800,
                                "create_response": True,
                                "interrupt_response": True,
                            },
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "voice": voice,
                        },
                    },
                },
            }))

            t0 = time.perf_counter()

            async def _sender() -> None:
                async for frame in media_iter:
                    if abort_event is not None and abort_event.is_set():
                        break
                    if frame.get("type") != "audio":
                        continue   # flush frames are no-op under server_vad
                    payload = frame.get("payload")
                    if not payload:
                        continue
                    pcm_16k = np.frombuffer(bytes(payload), dtype="int16")
                    pcm_24k = resample_poly(pcm_16k, 3, 2).astype("int16").tobytes()
                    for i in range(0, len(pcm_24k), _CHUNK):
                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm_24k[i: i + _CHUNK]).decode(),
                        }))

            sender_task = asyncio.create_task(_sender(), name="openai-realtime-translate-sender")
            try:
                while True:
                    if abort_event is not None and abort_event.is_set():
                        break
                    try:
                        msg = json.loads(await ws.recv())
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as exc:
                        yield {"type": "done", "error": f"{type(exc).__name__}: {exc}",
                               "latency_ms": 0.0, "cost_usd": 0.0}
                        break
                    t = msg.get("type", "")
                    if t == "response.output_audio.delta":
                        yield {
                            "type": "audio_b64",
                            "data": msg.get("delta", ""),
                            "sample_rate": 24000,
                        }
                    elif t == "response.output_audio_transcript.delta":
                        yield {
                            "type": "transcript",
                            "text": msg.get("delta", ""),
                            "is_final": False,
                        }
                    elif t == "response.done":
                        elapsed = time.perf_counter() - t0
                        yield {
                            "type": "done",
                            "latency_ms": elapsed * 1000,
                            "cost_usd": elapsed / 60.0 * (self.cost_per_call_estimate_usd or 0.034),
                        }
                        t0 = time.perf_counter()
                    elif t == "error":
                        log.warning("translate: openai error: %s",
                                    json.dumps(msg.get("error", {}))[:300])
            finally:
                sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await sender_task
