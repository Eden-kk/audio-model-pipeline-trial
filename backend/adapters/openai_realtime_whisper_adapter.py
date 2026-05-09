"""OpenAI Realtime Whisper ASR adapter — supports both batch and streaming.

Batch  : Open a Realtime WSS session, pipe the full audio as
         input_audio_buffer.append chunks, commit, request a response, and
         wait for response.done to collect the full transcript.
Stream : Same connection setup but yields partial transcript deltas as they
         arrive, with a final yield carrying the assembled text.

The Realtime API requires 24 kHz mono PCM16; we resample on the fly when the
source file has a different sample rate.

Env: OPENAI_API_KEY
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np
import soundfile as sf
import websockets  # type: ignore[import]

log = logging.getLogger("trial-app.openai_realtime_whisper")

_WS_URL = "wss://api.openai.com/v1/realtime?model={model}"
_CHUNK = 4096  # bytes per input_audio_buffer.append chunk


class OpenAIRealtimeWhisperAdapter:
    id = "openai_realtime_whisper"
    category = "asr"
    display_name = "OpenAI Realtime Whisper"
    hosting = "cloud"
    vendor = "OpenAI"
    is_streaming = True
    supported_languages: List[str] = [
        "auto", "en", "zh", "es", "fr", "de", "ja", "ko", "pt", "it",
        "nl", "pl", "ru", "tr", "vi", "ar", "hi", "id", "sv", "da",
        "fi", "nb", "cs", "sk", "ro", "hu", "el", "uk", "ca", "hr",
        "bg", "ms", "th", "lt", "lv", "et", "sl",
    ]
    multilang_realtime = False
    cost_per_call_estimate_usd: Optional[float] = None

    inputs: List[Dict[str, str]] = [{"name": "audio", "type": "audio_file"}]
    outputs: List[Dict[str, str]] = [{"name": "text", "type": "text"}]

    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "default": "auto",
                "description": "BCP-47 code or 'auto'",
            },
            "model": {
                "type": "string",
                "default": "gpt-realtime-mini",
                "description": "WSS-connection realtime carrier (GA endpoint)",
            },
            "transcription_model": {
                "type": "string",
                "default": "gpt-realtime-whisper",
                "enum": [
                    "gpt-realtime-whisper",
                    "gpt-4o-transcribe",
                    "gpt-4o-mini-transcribe",
                    "whisper-1",
                ],
                "description": "ASR model used for input_audio_transcription",
            },
            "timeout_s": {
                "type": "number",
                "default": 30,
            },
        },
    }

    def __init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            log.warning(
                "OPENAI_API_KEY not set — openai_realtime_whisper adapter will fail on first call"
            )

    def _api_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return key

    def _session_update(self, transcription_model: str) -> dict:
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": transcription_model},
                        "turn_detection": None,
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                    },
                },
            },
        }

    def _resample_to_24k(self, pcm: np.ndarray, src_sr: int) -> np.ndarray:
        if src_sr == 24000:
            return pcm
        from math import gcd
        from scipy.signal import resample_poly  # type: ignore[import]
        g = gcd(24000, src_sr)
        return resample_poly(pcm, 24000 // g, src_sr // g).astype("int16")

    async def transcribe(self, audio_path: str, config: dict) -> dict:
        """Run ASR via OpenAI Realtime WSS and return a normalised result dict."""
        language = config.get("language", "auto")
        model = config.get("model", "gpt-realtime-mini")
        transcription_model = config.get("transcription_model", "gpt-realtime-whisper")
        timeout_s = float(config.get("timeout_s", 30.0))
        api_key = self._api_key()

        data, sr = sf.read(audio_path, dtype="int16", always_2d=False)
        if data.ndim == 2:
            data = data[:, 0]
        data = self._resample_to_24k(data, sr)
        pcm_bytes = data.tobytes()
        duration_s = len(pcm_bytes) / (2 * 24000)

        t0 = time.perf_counter()

        async with websockets.connect(
            _WS_URL.format(model=model),
            additional_headers={"Authorization": f"Bearer {api_key}"},
            ping_interval=20,
        ) as ws:
            await ws.send(json.dumps(self._session_update(transcription_model)))

            for i in range(0, len(pcm_bytes), _CHUNK):
                chunk = pcm_bytes[i: i + _CHUNK]
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode(),
                }))

            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            async def _receive():
                full_text = ""
                raw_done = None
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    t = msg.get("type", "")
                    if t == "conversation.item.input_audio_transcription.delta":
                        full_text += msg.get("delta", "")
                    elif t == "conversation.item.input_audio_transcription.completed":
                        full_text = msg.get("transcript", full_text)
                        raw_done = msg
                        break
                return full_text, raw_done

            full_text, raw_done = await asyncio.wait_for(_receive(), timeout=timeout_s)

        wall_s = time.perf_counter() - t0
        return {
            "text": full_text,
            "words": [],
            "language": language,
            "duration_s": duration_s,
            "cost_usd": None,
            "wall_time_s": wall_s,
            "raw_response": raw_done,
        }

    async def transcribe_stream(
        self, audio_path: str, config: dict
    ) -> AsyncIterator[Dict[str, Any]]:
        language = config.get("language", "auto")
        model = config.get("model", "gpt-realtime-mini")
        transcription_model = config.get("transcription_model", "gpt-realtime-whisper")
        api_key = self._api_key()

        data, sr = sf.read(audio_path, dtype="int16", always_2d=False)
        if data.ndim == 2:
            data = data[:, 0]
        data = self._resample_to_24k(data, sr)
        pcm_bytes = data.tobytes()
        duration_s = len(pcm_bytes) / (2 * 24000)

        t0 = time.perf_counter()
        full_text = ""

        async with websockets.connect(
            _WS_URL.format(model=model),
            additional_headers={"Authorization": f"Bearer {api_key}"},
            ping_interval=20,
        ) as ws:
            await ws.send(json.dumps(self._session_update(transcription_model)))

            for i in range(0, len(pcm_bytes), _CHUNK):
                chunk = pcm_bytes[i: i + _CHUNK]
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode(),
                }))

            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = msg.get("type", "")
                if t == "conversation.item.input_audio_transcription.delta":
                    full_text += msg.get("delta", "")
                    yield {"partial_text": full_text, "is_final": False}
                elif t == "conversation.item.input_audio_transcription.completed":
                    full_text = msg.get("transcript", full_text)
                    break

        wall_s = time.perf_counter() - t0
        yield {
            "partial_text": full_text,
            "is_final": True,
            "text": full_text,
            "words": [],
            "language": language,
            "duration_s": duration_s,
            "cost_usd": None,
            "wall_time_s": wall_s,
        }
