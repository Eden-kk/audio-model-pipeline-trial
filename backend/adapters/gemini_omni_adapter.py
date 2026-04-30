"""Gemini Live realtime omni adapter — Vertex AI native-audio model.

Cloud counterpart to MiniCPMOAdapter. Implements the same `realtime_omni`
adapter contract (one `omni_session()` async generator yielding
transcript / text_delta / audio_b64 / done events) but talks to
`client.aio.live.connect()` from the unified google-genai SDK over a
persistent WebSocket.

Auth flows through Application Default Credentials. Set
`GOOGLE_APPLICATION_CREDENTIALS=/path/to/application_default_credentials.json`
in backend/.env; uvicorn picks it up via python-dotenv. There is no
fallback API-key path here — the iOS Live route uses the Generative
Language API + GEMINI_API_KEY and is wired separately (see docs/PLAN.md
P2 Slice 5).

Wire shape MUST match MiniCPMOAdapter so the existing Realtime UI
renders both adapters without per-adapter conditionals — in particular:

  * `audio_b64` events use key **`data`** (not `audio_b64`); the frontend
    reads `parsed.data` at `wsAudio.ts:227`.
  * The adapter MUST NOT emit `OmniReady` — the proxy emits it at
    `omni_proxy.py:69-74` before this generator is even invoked.
  * External config key is `max_new_tokens` (mirrors MiniCPM-o's slider);
    translated to Gemini's `max_output_tokens` at the SDK boundary.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional


log = logging.getLogger("trial-app.gemini_live")


class GeminiOmniAdapter:
    """realtime_omni adapter — Gemini Live 2.5 (Vertex AI native audio)."""

    id = "gemini_live"
    category = "realtime_omni"
    display_name = "Gemini Live 2.5 (Vertex, native audio)"
    hosting = "cloud"
    vendor = "google"
    is_streaming = True

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
                    "You are a helpful AR-glasses voice assistant. "
                    "Reply concisely and conversationally. "
                    "The [context] block below (if present) tells you who is "
                    "speaking; use it for awareness only — do not output it."
                ),
                "description": "Sent as Gemini's system_instruction on connect.",
            },
            # External name mirrors MiniCPM-o (the Realtime page's slider keys
            # off `max_new_tokens`). Translated to Gemini's `max_output_tokens`
            # at the SDK boundary inside omni_session.
            "max_new_tokens": {"type": "integer", "default": 256, "minimum": 16, "maximum": 1024},
            "temperature": {"type": "number", "default": 0.3, "minimum": 0.0, "maximum": 1.5},
            "generate_audio": {
                "type": "boolean", "default": True,
                "description": (
                    "When true, response_modalities=[AUDIO]. The "
                    "gemini-live-2.5-flash-native-audio model REQUIRES "
                    "AUDIO output — flipping this off triggers a Vertex "
                    "1007 'Text output is not supported for native audio "
                    "output model.' Override `model` if you need text-only."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Override the Live model id. Defaults to env "
                    "GEMINI_LIVE_MODEL, fallback gemini-live-2.5-flash-native-audio."
                ),
            },
        },
    }
    # Public Vertex pricing for Live native-audio is in the same bucket as
    # 2.5-flash; keep a coarse per-turn estimate so the UI's cost dial isn't
    # zero. Refine when we have invoice data.
    cost_per_call_estimate_usd: Optional[float] = 0.01

    def __init__(self) -> None:
        self._project = os.environ.get("GEMINI_VERTEX_PROJECT", "").strip()
        self._location = os.environ.get("GEMINI_VERTEX_LOCATION", "").strip() or "us-central1"
        self._default_model = (
            os.environ.get("GEMINI_LIVE_MODEL", "").strip()
            or "gemini-live-2.5-flash-native-audio"
        )
        if not self._project:
            log.warning(
                "GEMINI_VERTEX_PROJECT not set — gemini_live adapter will "
                "fail on first call. Set it in backend/.env."
            )

    async def omni_session(
        self,
        media_iter: AsyncIterator[Dict[str, Any]],
        *,
        config: Dict[str, Any],
        context_iter: Optional[AsyncIterator[str]] = None,
        abort_event: "Optional[asyncio.Event]" = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Drive a Vertex Live session. Receiver loop yields events; a
        sibling asyncio.Task pumps inbound media frames to the SDK.

        See plan-file note on lifecycle: this generator runs AFTER the
        proxy already emitted `OmniReady`, so we only yield domain
        events (transcript / text_delta / audio_b64 / done).
        """
        # Local imports keep backend cold-start light — the SDK pulls in
        # google-auth + grpc, which are sizeable.
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            yield {"type": "done",
                   "error": f"google-genai not installed: {e}",
                   "latency_ms": 0.0, "cost_usd": 0.0}
            return

        if not self._project:
            yield {"type": "done",
                   "error": "GEMINI_VERTEX_PROJECT not set",
                   "latency_ms": 0.0, "cost_usd": 0.0}
            return

        system_prompt = config.get("system_prompt") \
            or self.config_schema["properties"]["system_prompt"]["default"]
        max_new_tokens = int(config.get("max_new_tokens", 256))
        temperature = float(config.get("temperature", 0.3))
        generate_audio = bool(config.get("generate_audio", True))
        model = (config.get("model") or self._default_model).strip()

        client = genai.Client(
            vertexai=True, project=self._project, location=self._location,
        )
        live_config = types.LiveConnectConfig(
            response_modalities=[
                types.Modality.AUDIO if generate_audio else types.Modality.TEXT,
            ],
            system_instruction=system_prompt,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            generation_config=types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_new_tokens,
            ),
        )

        try:
            async with client.aio.live.connect(model=model, config=live_config) as session:
                async for ev in self._drive_session(
                    session, media_iter,
                    abort_event=abort_event,
                    types_module=types,
                ):
                    yield ev
        except Exception as e:
            # Connection-level failure (auth, model not found, etc.) —
            # surface as a `done` so the UI shows a clean error instead
            # of a silent WS close. The proxy's outer `except Exception`
            # at omni_proxy.py:259 will also emit OmniError for the same
            # case, but yielding `done` here is friendlier for the
            # adapter's own dial / cost panel.
            yield {"type": "done",
                   "error": f"{type(e).__name__}: {e}",
                   "latency_ms": 0.0, "cost_usd": 0.0}

    async def _drive_session(
        self,
        session: Any,
        media_iter: AsyncIterator[Dict[str, Any]],
        *,
        abort_event: "Optional[asyncio.Event]" = None,
        types_module: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Sender Task + receiver loop sharing one Live session.

        Sender failures are captured into `sender_exc` and surfaced as a
        `done` event with `error` on the next `turn_complete`. The
        `finally` cancels the sender — yields from `finally` after a
        propagating exception are silently lost (proxy already caught
        at omni_proxy.py:259), so cleanup only.
        """
        types = types_module
        t0 = time.perf_counter()
        sender_exc: List[BaseException] = []

        async def _sender() -> None:
            try:
                last_video_ts_ms = -10_000
                async for frame in media_iter:
                    if abort_event is not None and abort_event.is_set():
                        break
                    ftype = frame.get("type")
                    payload = frame.get("payload") or b""
                    if ftype == "audio":
                        if not payload:
                            continue
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=payload,
                                mime_type="audio/pcm;rate=16000",
                            ),
                        )
                    elif ftype == "video":
                        # Vertex Live throttles video at ~1 FPS. Drop
                        # client-side; ts_ms is proxy wall-clock, not
                        # time.monotonic(), but the delta is still valid.
                        # Throttle anchored on last *successful* send: skips
                        # don't shift the window.
                        if not payload:
                            continue
                        ts_ms = int(frame.get("ts_ms") or 0)
                        if ts_ms - last_video_ts_ms < 1000:
                            continue
                        await session.send_realtime_input(
                            video=types.Blob(
                                data=payload, mime_type="image/jpeg",
                            ),
                        )
                        last_video_ts_ms = ts_ms
                    elif ftype == "flush":
                        await session.send_realtime_input(audio_stream_end=True)
                    # Unknown types: silently skip to stay forward-compatible.
            except asyncio.CancelledError:
                raise
            except Exception as e:        # noqa: BLE001
                sender_exc.append(e)

        sender_task = asyncio.create_task(_sender(), name="gemini-sender")
        aborted = False
        any_turn_completed = False
        try:
            # Drive the receive loop by calling session._receive() directly.
            #
            # WHY NOT session.receive():
            #   The SDK's `session.receive()` is a single-turn iterator —
            #   it yields messages until it sees `turn_complete`, then
            #   `break`s. If we do `async for resp in session.receive():`
            #   the loop exits after the FIRST turn, exhausting this
            #   generator and causing _pump_adapter_to_browser to return,
            #   which in turn satisfies the proxy's FIRST_COMPLETED and
            #   tears the whole session down. A second utterance then has
            #   no live WS to talk to.
            #
            # WHY session._receive() IS SAFE:
            #   _receive() is a thin wrapper: await self._ws.recv() +
            #   JSON parse. It raises ConnectionClosed when the server
            #   closes the WS (session expiry, error, etc.), which we
            #   catch below and exit cleanly. No hidden state is kept
            #   between calls.
            while True:
                if abort_event is not None and abort_event.is_set():
                    aborted = True
                    break
                try:
                    resp = await session._receive()
                except Exception:
                    # ConnectionClosed or any other SDK error — server
                    # closed the live session (timeout, quota, etc.).
                    # Exit the loop; if we had a clean turn the UI
                    # already has its done; if not, fall through to the
                    # aborted path below.
                    break
                sc = getattr(resp, "server_content", None)

                # Transcription events — both Optional. Direct .text on a
                # None housekeeping event raises AttributeError and tears
                # down the session.
                if sc is not None:
                    in_tx = getattr(sc, "input_transcription", None)
                    if in_tx is not None and getattr(in_tx, "text", None):
                        yield {"type": "transcript",
                               "text": in_tx.text,
                               "is_final": bool(getattr(in_tx, "finished", False))}
                    out_tx = getattr(sc, "output_transcription", None)
                    if out_tx is not None and getattr(out_tx, "text", None):
                        yield {"type": "text_delta", "text": out_tx.text}

                # Audio bytes — `LiveServerMessage.data` is a convenience
                # property over server_content.model_turn.parts[i].inline_data.data.
                # Native audio output is 24 kHz Int16 PCM. The frontend's
                # decodeAudioData fallback at wsAudio.ts:163 honors the
                # event's `sample_rate` field.
                if getattr(resp, "data", None):
                    yield {"type": "audio_b64",
                           "data": base64.b64encode(resp.data).decode("ascii"),
                           "sample_rate": 24000}

                if sc is not None and getattr(sc, "turn_complete", False):
                    # Per-turn done. Stay in the receive loop — Gemini
                    # Live keeps the WS open across turns; we continue
                    # `while True` here so the next utterance's audio
                    # frames (already queued by the sender task) flow
                    # into the same session without a reconnect.
                    if sender_exc:
                        yield {"type": "done",
                               "error": f"{type(sender_exc[0]).__name__}: {sender_exc[0]}",
                               "latency_ms": 0.0, "cost_usd": 0.0}
                        sender_exc.clear()
                    else:
                        yield {"type": "done",
                               "latency_ms": (time.perf_counter() - t0) * 1000.0,
                               "cost_usd": self.cost_per_call_estimate_usd or 0.0}
                    any_turn_completed = True
                    t0 = time.perf_counter()       # baseline for next turn

            # Abort path — receiver broke without an in-flight turn. If
            # at least one turn already completed cleanly, the UI has its
            # done; suppress the duplicate. Mirrors MiniCPM-o's
            # aborted=True done at minicpm_o_adapter.py:362-365.
            if aborted and not any_turn_completed:
                yield {"type": "done",
                       "latency_ms": (time.perf_counter() - t0) * 1000.0,
                       "cost_usd": 0.0,
                       "aborted": True}
        finally:
            sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender_task
