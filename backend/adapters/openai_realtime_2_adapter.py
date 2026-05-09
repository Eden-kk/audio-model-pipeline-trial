"""OpenAI Realtime 2 adapter — GPT-5-class bidirectional voice AI.

Uses OpenAI's GA Realtime API (no `OpenAI-Beta` header). Persistent WSS
session with while-True multi-turn loop. abort_event handled by receiver
(not sender). Env: OPENAI_API_KEY
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

log = logging.getLogger("trial-app.openai_realtime_2")

_WS_URL = "wss://api.openai.com/v1/realtime?model={model}"
_CHUNK = 4096


class OpenAIRealtime2Adapter:
    id = "openai_realtime_2"
    category = "realtime_omni"
    display_name = "OpenAI Realtime 2"
    hosting = "cloud"
    vendor = "OpenAI"
    is_streaming = True
    cost_per_call_estimate_usd: Optional[float] = 0.05  # flat per-turn estimate; refine when per-minute pricing is confirmed

    inputs: List[Dict[str, str]] = [{"name": "media", "type": "media_stream"}]
    outputs: List[Dict[str, str]] = [{"name": "events", "type": "omni_event"}]

    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "system_prompt": {
                "type": "string",
                "default": "You are a helpful AR-glasses voice assistant. Reply concisely.",
            },
            "reasoning_effort": {
                "type": "string",
                "default": "medium",
                "enum": ["minimal", "low", "medium", "high", "xhigh"],
                "description": "GA: sent as response.reasoning.effort, not in session.update",
            },
            "voice": {
                "type": "string",
                "default": "alloy",
                "enum": ["alloy", "echo", "shimmer", "fable", "onyx", "nova"],
            },
            "max_new_tokens": {
                "type": "integer",
                "default": 256,
                "minimum": 16,
                "maximum": 2048,
            },
            "generate_audio": {
                "type": "boolean",
                "default": True,
            },
            "model": {
                "type": "string",
                "default": "gpt-realtime-2",
            },
        },
    }

    def __init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            log.warning(
                "OPENAI_API_KEY not set — openai_realtime_2 adapter will fail on first call"
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
        system_prompt = config.get(
            "system_prompt",
            "You are a helpful AR-glasses voice assistant. Reply concisely.",
        )
        reasoning_effort = config.get("reasoning_effort", "medium")
        voice = config.get("voice", "alloy")
        max_new_tokens = int(config.get("max_new_tokens", 256))
        generate_audio = bool(config.get("generate_audio", True))
        model = config.get("model", "gpt-realtime-2")
        api_key = self._api_key()

        # GA only supports a single output modality per response: ['audio'] or ['text'].
        output_modalities = ["audio"] if generate_audio else ["text"]

        async with websockets.connect(
            _WS_URL.format(model=model),
            additional_headers={"Authorization": f"Bearer {api_key}"},
            ping_interval=20,
        ) as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": system_prompt,
                    "max_output_tokens": max_new_tokens,
                    "output_modalities": output_modalities,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": "whisper-1"},
                            "turn_detection": None,
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "voice": voice,
                        },
                    },
                },
            }))

            t0 = time.perf_counter()

            async def _context_drainer() -> None:
                if context_iter is None:
                    return
                async for chunk in context_iter:
                    line = str(chunk).strip()
                    if not line:
                        continue
                    try:
                        await ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "system",
                                "content": [{"type": "input_text", "text": line}],
                            },
                        }))
                    except websockets.exceptions.ConnectionClosed:
                        return

            async def _sender() -> None:
                async for frame in media_iter:
                    if abort_event is not None and abort_event.is_set():
                        break
                    ftype = frame.get("type")
                    if ftype == "audio":
                        payload = frame.get("payload")
                        if not payload:
                            continue
                        pcm_16k = np.frombuffer(bytes(payload), dtype="int16")
                        pcm_24k = resample_poly(pcm_16k, 3, 2).astype("int16")
                        pcm_bytes = pcm_24k.tobytes()
                        for i in range(0, len(pcm_bytes), _CHUNK):
                            await ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(pcm_bytes[i: i + _CHUNK]).decode(),
                            }))
                    elif ftype == "flush":
                        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        await ws.send(json.dumps({
                            "type": "response.create",
                            "response": {
                                "reasoning": {"effort": reasoning_effort},
                                "output_modalities": output_modalities,
                            },
                        }))
                    # unknown/video frames: silently skip (forward-compat)

            sender_task = asyncio.create_task(_sender(), name="openai-realtime-2-sender")
            context_task = asyncio.create_task(_context_drainer(), name="openai-realtime-2-context")
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
                    if t == "conversation.item.input_audio_transcription.delta":
                        yield {
                            "type": "transcript",
                            "text": msg.get("delta", ""),
                            "is_final": False,
                        }
                    elif t in ("response.output_text.delta",
                               "response.output_audio_transcript.delta"):
                        yield {"type": "text_delta", "text": msg.get("delta", "")}
                    elif t == "response.output_audio.delta":
                        yield {
                            "type": "audio_b64",
                            "data": msg.get("delta", ""),
                            "sample_rate": 24000,
                        }
                    elif t == "response.function_call_arguments.done":
                        raw_args = msg.get("arguments", "{}")
                        try:
                            parsed_args = json.loads(raw_args)
                        except (json.JSONDecodeError, TypeError):
                            parsed_args = {"_raw": raw_args}
                        yield {
                            "type": "tool_call",
                            "name": msg.get("name", ""),
                            "args": parsed_args,
                        }
                    elif t == "response.done":
                        elapsed = time.perf_counter() - t0
                        yield {
                            "type": "done",
                            "latency_ms": elapsed * 1000.0,
                            "cost_usd": self.cost_per_call_estimate_usd or 0.0,
                        }
                        t0 = time.perf_counter()
            finally:
                sender_task.cancel()
                context_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender_task
                with contextlib.suppress(asyncio.CancelledError):
                    await context_task
