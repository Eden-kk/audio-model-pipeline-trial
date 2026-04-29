"""MiniCPM-o realtime omni adapter — talks to model-server/modal_app_minicpm.py.

This is the **edge candidate** for the fast-loop bench (vs Gemini Live's
cloud candidate, when that lands). Implements the `realtime_omni` adapter
protocol via **per-utterance chunked HTTP**:

  1. Browser opens /ws/omni; the proxy hands us an async iter of inbound
     media frames ({type='audio'|'video'|'flush', payload, ts_ms}).
  2. We accumulate audio Int16 PCM (and the most-recent JPEG frame, if
     any) in memory until a `flush` arrives.
  3. On `flush`, we POST {audio_b64, image_b64?, system_prompt, history}
     to the Modal endpoint's /v1/omni and stream the response back as
     omni events (transcript / text_delta / audio_b64 / tool_call / done).

This is the v1 "TTFA degraded" path the plan calls out — full duplex with
sub-second TTFA needs a real WS shim on the Modal side, which is v1.5.
The HTTP path keeps the round-trip understandable: each user-turn is one
request, one response, easy to log + replay.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import struct
import time
from typing import Any, AsyncIterator, Dict, List, Optional


log = logging.getLogger("trial-app.minicpm_o")


def _int16_pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 16000,
                            channels: int = 1) -> bytes:
    """Wrap raw Int16LE PCM bytes in a minimal WAV header so MiniCPM's
    soundfile.read() (server-side) can decode it. ~44 bytes overhead.
    """
    n_samples = len(pcm) // 2
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    chunk_size = 36 + len(pcm)
    return (
        b"RIFF" + struct.pack("<I", chunk_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH",
                                16, 1, channels, sample_rate, byte_rate,
                                block_align, 16)
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    )


class MiniCPMOAdapter:
    """realtime_omni adapter — MiniCPM-o-4.5 on Modal A100-40GB."""

    id = "minicpm_o"
    category = "realtime_omni"
    display_name = "MiniCPM-o 4.5 (Modal A100, edge candidate)"
    hosting = "modal"
    vendor = "OpenBMB"
    is_streaming = True   # streams omni events; underlying transport is HTTP per turn

    inputs: List[Dict[str, str]] = [
        {"name": "media", "type": "media_stream"},
    ]
    outputs: List[Dict[str, str]] = [
        {"name": "events", "type": "omni_event"},
    ]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "system_prompt": {
                "type": "string",
                "default": (
                    "You are a helpful realtime voice assistant. Reply briefly. "
                    "When you see context lines like 'wearer just said: …' or "
                    "'stranger just said: …', use them to know who is speaking."
                ),
                "description": "Prepended to every utterance request.",
            },
            "max_new_tokens": {"type": "integer", "default": 256, "minimum": 16, "maximum": 1024},
            "temperature": {"type": "number", "default": 0.3, "minimum": 0.0, "maximum": 1.5},
            "generate_audio": {
                "type": "boolean", "default": True,
                "description": "When true, response includes a synthesised audio reply (slower TTFA, full TTS render).",
            },
            "url": {
                "type": "string",
                "description": "Override the deployed Modal endpoint. Defaults to env MINICPM_O_REALTIME_URL.",
            },
            "request_timeout_s": {"type": "number", "default": 120.0},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = 0.05   # ~A100-min/turn warm

    def __init__(self) -> None:
        self._url = os.environ.get("MINICPM_O_REALTIME_URL", "").rstrip("/")
        if not self._url:
            log.warning(
                "MINICPM_O_REALTIME_URL not set — minicpm_o adapter will fail "
                "on first call. Set it to the deployed Modal app URL."
            )

    # The adapter doesn't implement transcribe/synthesize/etc — it ONLY
    # implements omni_session. The runner / /ws/omni proxy dispatches by
    # category and won't call the wrong method.

    async def omni_session(
        self,
        media_iter: AsyncIterator[Dict[str, Any]],
        *,
        config: Dict[str, Any],
        context_iter: Optional[AsyncIterator[str]] = None,
        abort_event: "Optional[asyncio.Event]" = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming omni session against /v1/omni-stream (Slice O6.2).

        Consumes NDJSON events from the Modal endpoint and yields them
        through to the proxy/browser. Falls back to the older /v1/omni
        blocking path when `config.use_streaming = False` — useful as an
        operator escape hatch if the streaming endpoint misbehaves.

        `abort_event` (Slice O6.2): when set, the inner stream loop exits
        and the in-flight httpx response context is closed. The proxy's
        `interrupt` event flips this so a user starting to talk mid-reply
        can cancel the model's audio playback cleanly.
        """
        import asyncio   # noqa: WPS433
        import httpx     # local import — keeps backend cold-import fast
        import json as _json

        url = (config.get("url") or self._url).rstrip("/")
        if not url:
            yield {"type": "done", "error": "MINICPM_O_REALTIME_URL not set",
                   "latency_ms": 0.0, "cost_usd": 0.0}
            return

        system_prompt = config.get("system_prompt") or self.config_schema["properties"]["system_prompt"]["default"]
        max_new_tokens = int(config.get("max_new_tokens", 256))
        temperature = float(config.get("temperature", 0.3))
        generate_audio = bool(config.get("generate_audio", True))
        request_timeout_s = float(config.get("request_timeout_s", 120.0))
        # Default to streaming endpoint; flip to False to fall back to the
        # blocking /v1/omni path when needed (e.g. for diagnosing whether
        # the streaming endpoint introduced a regression).
        use_streaming = bool(config.get("use_streaming", True))

        # Per-session rolling state
        pcm_buffer = bytearray()              # all audio since the last flush
        latest_image: Optional[bytes] = None  # most-recent JPEG frame
        history: List[Dict[str, Any]] = []    # prior turns (text-only — audio history is too heavy for HTTP path)
        context_lines: List[str] = []         # wearer-tag heartbeat injections

        async def _drain_context() -> None:
            """Pull any pending strings from context_iter without blocking."""
            if context_iter is None:
                return
            # context_iter is expected to be a queue-backed async iter; we
            # use a non-blocking poll via wait_for(0) rather than blocking
            # on the next yield.
            try:
                import asyncio
                while True:
                    line = await asyncio.wait_for(context_iter.__anext__(), timeout=0.001)
                    if line:
                        context_lines.append(line)
            except (StopAsyncIteration, Exception):
                pass

        async with httpx.AsyncClient(timeout=request_timeout_s) as client:
            async for frame in media_iter:
                ftype = frame.get("type")
                if ftype == "audio":
                    pcm_buffer.extend(frame.get("payload") or b"")
                    continue
                if ftype == "video":
                    latest_image = frame.get("payload")
                    continue
                if ftype != "flush":
                    # Unknown frame type — log + skip
                    log.warning(f"minicpm_o: unknown frame type {ftype!r}")
                    continue

                # ── End-of-utterance: assemble the request ──────────────
                if not pcm_buffer:
                    # Empty utterance (fast-fingered flush); just ack.
                    yield {"type": "done", "latency_ms": 0.0, "cost_usd": 0.0,
                           "note": "empty utterance"}
                    continue

                await _drain_context()
                wav_bytes = _int16_pcm_to_wav_bytes(bytes(pcm_buffer))
                pcm_buffer.clear()

                # Wearer-tag context appended as additional system lines
                # so the model has fresh "who's talking" hints.
                merged_system = system_prompt
                if context_lines:
                    merged_system = (
                        merged_system + "\n\n[recent context]\n"
                        + "\n".join(context_lines[-8:])     # cap at last 8 lines
                    )

                body: Dict[str, Any] = {
                    "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                    "system_prompt": merged_system,
                    "history": history,
                    "generate_audio": generate_audio,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }
                if latest_image is not None:
                    body["image_b64"] = base64.b64encode(latest_image).decode("ascii")

                # Emit a synthetic transcript event so the UI's caption strip
                # has something to show before the response arrives.
                yield {"type": "transcript",
                       "text": "(processing utterance…)",
                       "is_final": False}

                t0 = time.perf_counter()

                # ── Streaming path: NDJSON over StreamingResponse ────────
                # Consume one JSON object per line from /v1/omni-stream.
                # Per-line aborts are checked against `abort_event`; when
                # the proxy flips it, we close the response context (httpx
                # cancels the underlying TCP read).
                if use_streaming:
                    aggregated_text = ""
                    aborted = False
                    final_latency_ms: Optional[float] = None
                    try:
                        async with client.stream(
                            "POST", f"{url}/v1/omni-stream", json=body,
                        ) as resp:
                            if resp.status_code != 200:
                                # Drain body for the error message.
                                err_body = (await resp.aread()).decode("utf-8", "replace")[:200]
                                yield {"type": "done",
                                       "error": f"HTTP {resp.status_code}: {err_body}",
                                       "latency_ms": (time.perf_counter() - t0) * 1000.0,
                                       "cost_usd": 0.0}
                                continue
                            async for line in resp.aiter_lines():
                                if abort_event is not None and abort_event.is_set():
                                    aborted = True
                                    break
                                if not line:
                                    continue
                                try:
                                    ev = _json.loads(line)
                                except _json.JSONDecodeError:
                                    log.warning(f"minicpm_o: malformed NDJSON line: {line[:80]!r}")
                                    continue
                                ev_type = ev.get("type")
                                if ev_type == "transcript":
                                    aggregated_text = ev.get("text") or aggregated_text
                                    yield ev
                                elif ev_type in ("text_delta", "audio_b64", "tool_call"):
                                    yield ev
                                elif ev_type == "done":
                                    final_latency_ms = float(ev.get("latency_ms", 0.0))
                                    streaming_path = ev.get("streaming_path", "?")
                                    # Don't re-emit done from the upstream — we
                                    # synthesize our own done with our cost
                                    # estimate after the stream closes.
                                    yield {"type": "meta",
                                           "streaming_path": streaming_path}
                                # Unknown types: silently passthrough so future
                                # event additions on the Modal side don't break
                                # the adapter.
                                else:
                                    yield ev
                    except httpx.RequestError as e:
                        yield {"type": "done",
                               "error": f"network: {type(e).__name__}: {e}",
                               "latency_ms": (time.perf_counter() - t0) * 1000.0,
                               "cost_usd": 0.0}
                        continue
                    except asyncio.CancelledError:
                        # Proxy cancelled us — propagate without yielding.
                        raise

                    # Successful (or aborted) stream completion. Update
                    # history with the aggregated assistant text.
                    if aggregated_text:
                        history.append({"role": "assistant", "content": [aggregated_text]})
                        if len(history) > 12:
                            history = history[-12:]
                    if abort_event is not None:
                        abort_event.clear()    # ready for next utterance

                    latency_ms = final_latency_ms if final_latency_ms is not None \
                        else (time.perf_counter() - t0) * 1000.0
                    yield {"type": "done",
                           "latency_ms": latency_ms,
                           "cost_usd": self.cost_per_call_estimate_usd or 0.0,
                           "aborted": aborted}
                    continue

                # ── Fallback path: blocking /v1/omni (operator escape hatch)
                try:
                    resp = await client.post(f"{url}/v1/omni", json=body)
                except httpx.RequestError as e:
                    yield {"type": "done", "error": f"network: {type(e).__name__}: {e}",
                           "latency_ms": (time.perf_counter() - t0) * 1000.0,
                           "cost_usd": 0.0}
                    continue
                if resp.status_code != 200:
                    yield {"type": "done",
                           "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                           "latency_ms": (time.perf_counter() - t0) * 1000.0,
                           "cost_usd": 0.0}
                    continue
                data = resp.json()
                text = data.get("text") or ""
                audio_b64_out = data.get("audio_b64")
                sample_rate = data.get("sample_rate") or 16000
                if text:
                    yield {"type": "transcript", "text": text, "is_final": True}
                    yield {"type": "text_delta", "text": text}
                if audio_b64_out:
                    yield {"type": "audio_b64", "data": audio_b64_out,
                           "sample_rate": sample_rate}
                history.append({"role": "assistant", "content": [text]})
                if len(history) > 12:
                    history = history[-12:]
                latency_ms = float(data.get("latency_ms", (time.perf_counter() - t0) * 1000.0))
                yield {"type": "done",
                       "latency_ms": latency_ms,
                       "cost_usd": self.cost_per_call_estimate_usd or 0.0}
