"""Browser ↔ realtime_omni adapter WebSocket proxy.

Wires the trial-app frontend's `/ws/omni?adapter=…` route into the
adapter's `omni_session()` async generator. Three coroutines fan out:

  1. `_pump_browser_to_adapter` — reads binary frames from the client WS,
     demuxes the 1-byte type tag (0x01 audio PCM · 0x02 JPEG video ·
     0x03 flush) into a structured `{type, payload, ts_ms}` dict, and
     pushes onto an asyncio.Queue that the adapter consumes.

  2. `_pump_adapter_to_browser` — drains the adapter's omni_event stream
     and forwards each event to the client WS as JSON.

  3. (Slice O5) `_pump_wearer_context` — periodic pyannote_verify pass on
     the most-recent audio buffer, pushing context strings onto a separate
     queue the adapter reads via `context_iter`. Stubbed in v1; real
     implementation lands with the wearer-tag heartbeat slice.

Binary protocol: every inbound WS frame is `[type:u8][payload:bytes]`.
This keeps the worklet simple — it just prepends `0x01` to every PCM
chunk. JPEG frames will use `0x02` in Slice O3. `0x03` is a 1-byte
end-of-utterance marker the page sends on push-to-talk release.

Outbound is plain JSON per omni event (the adapter's contract).
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect


log = logging.getLogger("trial-app.omni-proxy")


# Binary frame type tags — matches the discriminator the browser worklet emits.
FRAME_AUDIO = 0x01
FRAME_VIDEO = 0x02
FRAME_FLUSH = 0x03


async def proxy_omni_session(
    client_ws: WebSocket,
    adapter_obj: Any,
    *,
    config: Dict[str, Any],
    wearer_adapter: Any = None,
    wearer_embedding_b64: Optional[str] = None,
) -> None:
    """Run a single end-to-end omni session for one connected browser.

    Lifecycle:
      - Client connects to /ws/omni?adapter=<id>; the route handler in
        main.py validates the adapter is `realtime_omni`, accepts the WS,
        and invokes us.
      - We open two queues: media_in (browser→adapter) and context_in
        (heartbeat→adapter, populated by _pump_wearer_context if a
        wearer profile is enrolled).
      - We spawn 2-3 tasks: browser-pump, adapter-pump, and (when wearer
        is enrolled) wearer-context heartbeat.
      - When any task finishes (client disconnect / adapter done / error),
        we cancel the others and close the WS cleanly.
    """
    await client_ws.send_json({
        "event": "OmniReady",
        "adapter": getattr(adapter_obj, "id", "?"),
        "wearer_enrolled": bool(wearer_embedding_b64),
        "ts_ms": int(time.time() * 1000),
    })

    media_q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(maxsize=512)
    context_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=64)

    # Slice O6.2: shared abort_event lets the browser interrupt mid-reply.
    # Browser sends `{event: 'interrupt'}` → _pump_browser_to_adapter sets
    # this; the adapter checks it inside its NDJSON consumer loop.
    abort_event = asyncio.Event()

    # Rolling audio buffer the wearer-tag heartbeat reads from. Updated by
    # _pump_browser_to_adapter on every audio frame. Cap at 10 s to keep
    # memory bounded for long sessions.
    audio_buffer = bytearray()
    AUDIO_BUFFER_MAX_BYTES = 16000 * 2 * 10   # 10 s @ 16 kHz Int16 mono

    async def media_iter() -> AsyncIterator[Dict[str, Any]]:
        """Adapter consumes from this — translates None into termination."""
        while True:
            frame = await media_q.get()
            if frame is None:
                return
            yield frame

    async def context_iter() -> AsyncIterator[str]:
        while True:
            line = await context_q.get()
            if line is None:
                return
            yield line

    browser_task = asyncio.create_task(
        _pump_browser_to_adapter(
            client_ws, media_q, audio_buffer, AUDIO_BUFFER_MAX_BYTES,
            abort_event=abort_event,
        ),
        name="omni-browser-pump",
    )
    adapter_task = asyncio.create_task(
        _pump_adapter_to_browser(
            client_ws, adapter_obj, media_iter(), context_iter(), config,
            abort_event=abort_event,
        ),
        name="omni-adapter-pump",
    )
    tasks = {browser_task, adapter_task}

    if wearer_adapter is not None and wearer_embedding_b64:
        heartbeat_task = asyncio.create_task(
            _pump_wearer_context(
                client_ws, wearer_adapter, wearer_embedding_b64,
                audio_buffer, context_q,
            ),
            name="omni-wearer-heartbeat",
        )
        tasks.add(heartbeat_task)

    done, pending = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED,
    )
    # Whichever finished first, tear the other down + close the WS.
    for t in pending:
        t.cancel()
    for t in pending:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    # Surface errors on the failing task to logs (not to the user — at
    # this point the WS is already going away).
    for t in done:
        exc = t.exception()
        if exc and not isinstance(exc, (asyncio.CancelledError, WebSocketDisconnect)):
            log.warning(f"omni session task {t.get_name()} died: {type(exc).__name__}: {exc}")

    try:
        await client_ws.close()
    except Exception:
        pass


async def _pump_browser_to_adapter(
    client_ws: WebSocket,
    media_q: asyncio.Queue,
    audio_buffer: bytearray,
    audio_buffer_max_bytes: int,
    *,
    abort_event: Optional[asyncio.Event] = None,
) -> None:
    """Read tagged binary frames from the client; push structured dicts.

    Also accepts JSON text frames as a control channel (e.g. `{"event":
    "ping"}` or `{"event": "flush"}` for clients that prefer not to send
    a 1-byte binary).

    `audio_buffer` is a SHARED rolling buffer the wearer-tag heartbeat
    coroutine reads from. We append every PCM frame here AND push it
    onto media_q for the adapter; the buffer is independent of the
    adapter's per-utterance accumulator.
    """
    try:
        while True:
            msg = await client_ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                if not data:
                    continue
                tag = data[0]
                payload = data[1:]
                ts_ms = int(time.time() * 1000)
                if tag == FRAME_AUDIO:
                    await media_q.put({"type": "audio", "payload": payload, "ts_ms": ts_ms})
                    audio_buffer.extend(payload)
                    # Evict oldest bytes once we exceed the cap (rolling FIFO).
                    if len(audio_buffer) > audio_buffer_max_bytes:
                        del audio_buffer[: len(audio_buffer) - audio_buffer_max_bytes]
                elif tag == FRAME_VIDEO:
                    await media_q.put({"type": "video", "payload": payload, "ts_ms": ts_ms})
                elif tag == FRAME_FLUSH:
                    await media_q.put({"type": "flush", "payload": b"", "ts_ms": ts_ms})
                else:
                    log.warning(f"omni proxy: unknown binary frame tag 0x{tag:02x}; skipping")
            elif "text" in msg and msg["text"] is not None:
                # JSON control channel — keepalive + flush + future config tweaks.
                try:
                    payload = _json.loads(msg["text"])
                except _json.JSONDecodeError:
                    continue
                ev = payload.get("event")
                if ev == "ping":
                    await client_ws.send_json({"event": "pong"})
                elif ev == "flush":
                    await media_q.put({"type": "flush", "payload": b"",
                                       "ts_ms": int(time.time() * 1000)})
                elif ev == "interrupt":
                    # Slice O6.2: browser detected user-speech-start while
                    # the model was still replying. Flip the abort event
                    # so the adapter's NDJSON consumer exits its inner
                    # stream loop on the next iteration.
                    if abort_event is not None:
                        abort_event.set()
                elif ev == "stop":
                    # Client wants to end the session; signal the adapter.
                    break
                # Unknown events are silently ignored — keeps the protocol
                # forward-compatible with v1.5 client features.
    except WebSocketDisconnect:
        pass
    finally:
        # Always signal end-of-stream to the adapter so it can wrap up.
        await media_q.put(None)


async def _pump_adapter_to_browser(
    client_ws: WebSocket,
    adapter_obj: Any,
    media_iter: AsyncIterator[Dict[str, Any]],
    context_iter: AsyncIterator[str],
    config: Dict[str, Any],
    *,
    abort_event: Optional[asyncio.Event] = None,
) -> None:
    """Drive the adapter's omni_session and forward every event to the client.

    `abort_event` is plumbed through to the adapter so it can interrupt
    its in-flight NDJSON stream when the user starts speaking mid-reply.
    """
    try:
        async for ev in adapter_obj.omni_session(
            media_iter,
            config=config,
            context_iter=context_iter,
            abort_event=abort_event,
        ):
            # Tag every event with a server timestamp so the client can
            # measure TTFA from its local clock without round-trip skew.
            ev_out = {**ev, "server_ts_ms": int(time.time() * 1000)}
            try:
                await client_ws.send_json(ev_out)
            except (WebSocketDisconnect, RuntimeError):
                # Client gone mid-stream — stop pumping.
                return
    except Exception as e:
        # Surface adapter errors as a final OmniError event so the UI can
        # show "MiniCPM-o failed: <error>" instead of just disconnecting.
        log.warning(f"omni adapter loop died: {type(e).__name__}: {e}")
        try:
            await client_ws.send_json({
                "event": "OmniError",
                "error": f"{type(e).__name__}: {e}",
            })
        except Exception:
            pass


async def _pump_wearer_context(
    client_ws: WebSocket,
    wearer_adapter: Any,
    wearer_embedding_b64: str,
    audio_buffer: bytearray,
    context_q: asyncio.Queue,
    *,
    interval_s: float = 6.0,
    window_s: float = 6.0,
) -> None:
    """Periodic pyannote_verify pass on the rolling buffer; pushes
    speaker-awareness hints into context_q for the adapter's prompt.

    The heartbeat runs every `interval_s` seconds (default 6 s, doubled
    from the original 3 s to halve pyannote load on the event loop) and
    analyses the most-recent `window_s` seconds of audio.  If the buffer
    hasn't accumulated enough samples yet, we sleep through the tick.

    The pyannote verify call is dispatched as a background asyncio task so
    it never blocks the request path.  Results arrive asynchronously and
    are pushed onto context_q when ready.

    Also forwards a 'WearerTag' UI event to the client so the Realtime
    page can render the wearer-overlay (green/grey for last window).

    Failure mode: if the verify call errors (CPU starved, model crash, …)
    we log + continue.  Heartbeat misses are non-fatal.
    """
    import os
    import struct
    import tempfile

    BYTES_PER_SAMPLE = 2
    SAMPLE_RATE = 16000
    min_bytes = int(window_s * SAMPLE_RATE * BYTES_PER_SAMPLE)

    def _wrap_wav(pcm_bytes: bytes) -> bytes:
        byte_rate = SAMPLE_RATE * BYTES_PER_SAMPLE
        chunk_size = 36 + len(pcm_bytes)
        return (
            b"RIFF" + struct.pack("<I", chunk_size) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH",
                                    16, 1, 1, SAMPLE_RATE, byte_rate, 2, 16)
            + b"data" + struct.pack("<I", len(pcm_bytes)) + pcm_bytes
        )

    async def _verify_and_push(snapshot: bytes, wav_path: str) -> None:
        """Background task: run pyannote verify on a snapshot, then push
        the result onto context_q and send a WearerTag UI event.  Runs
        entirely off the request critical path — the omni POST is never
        serialised behind this call.
        """
        try:
            result = await wearer_adapter.verify_segments(
                wav_path,
                enrolled_embedding_b64=wearer_embedding_b64,
                config={"window_s": 1.0, "hop_s": 0.5, "threshold": 0.4},
            )
        except Exception as e:
            log.debug(f"wearer heartbeat verify failed: {type(e).__name__}: {e}")
            return
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        segments = result.get("segments") or []
        user_count = sum(1 for s in segments if s.get("is_user"))
        stranger_count = len(segments) - user_count

        # Build the context hint — biased toward the dominant speaker.
        # Use neutral phrasing ("speaking") rather than "just said: <X>"
        # to avoid priming the model to echo that format in its reply.
        if user_count > stranger_count:
            line = f"[speaker: wearer — {user_count}/{len(segments)} segments matched in last {window_s:.0f}s]"
        elif stranger_count > 0:
            line = f"[speaker: other — {stranger_count}/{len(segments)} segments unmatched in last {window_s:.0f}s]"
        else:
            return   # silence — don't spam the prompt

        # Best-effort push (drops if full)
        try:
            context_q.put_nowait(line)
        except asyncio.QueueFull:
            pass

        # UI-side event for the wearer overlay
        try:
            await client_ws.send_json({
                "event": "WearerTag",
                "user_segments": user_count,
                "stranger_segments": stranger_count,
                "n_segments": len(segments),
                "window_s": window_s,
            })
        except Exception:
            pass   # client disconnected — main loop will cancel us shortly

    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            break

        if len(audio_buffer) < min_bytes:
            continue

        # Snapshot the most-recent window_s of audio and write it to a
        # temp WAV.  The actual verify call runs in a background task so
        # the heartbeat loop returns immediately and never delays an omni
        # request that may be in-flight concurrently.
        snapshot = bytes(audio_buffer[-min_bytes:])
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(_wrap_wav(snapshot))
            tmp.close()
        except Exception as e:
            log.debug(f"wearer heartbeat WAV write failed: {type(e).__name__}: {e}")
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            continue

        # Fire-and-forget — do NOT await here.
        asyncio.create_task(
            _verify_and_push(snapshot, tmp.name),
            name="omni-wearer-verify",
        )
