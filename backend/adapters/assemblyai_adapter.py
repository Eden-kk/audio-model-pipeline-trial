"""AssemblyAI ASR adapter — supports both batch (Universal-2) and
streaming (Universal-Streaming v3).

Batch  : POST to /v2/upload + /v2/transcript, poll until done. Returns
         the full transcript at once.
Stream : Open a wss://streaming.assemblyai.com/v3/ws connection, pipe
         PCM s16le @ 16 kHz in real-time-paced chunks, receive Turn
         frames. Unformatted turns are partials; formatted turns are
         the cleaned-up final per turn.

Env: ASSEMBLYAI_API_KEY (works for both; static key — no temp-token dance).
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import time
import wave
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

_UPLOAD = "https://api.assemblyai.com/v2/upload"
_TRANSCRIPT = "https://api.assemblyai.com/v2/transcript"
_STREAMING_WS = "wss://streaming.assemblyai.com/v3/ws"


class AssemblyAIAdapter:
    id = "assemblyai"
    category = "asr"
    display_name = "AssemblyAI Universal-2 / Streaming"
    hosting = "cloud"
    vendor = "AssemblyAI"
    is_streaming = True   # transcribe_stream() implemented via Universal-Streaming v3

    inputs: List[Dict[str, str]] = [{"name": "audio", "type": "audio_file"}]
    outputs: List[Dict[str, str]] = [
        {"name": "text", "type": "text"},
        {"name": "words", "type": "word_timing"},
    ]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "speech_model": {"type": "string", "default": "universal-2",
                             "enum": ["universal-2", "universal-3-pro"],
                             "description": "Used by /v2/transcript (batch only)."},
            "streaming_model": {
                "type": "string",
                "default": "universal-streaming-english",
                "enum": ["universal-streaming-english", "universal-streaming-multilingual"],
                "description": "Used by Universal-Streaming v3 WS (when is_streaming).",
            },
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
        speech_model = config.get("speech_model", "universal-2")

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(audio_path, "rb") as f:
                r1 = await client.post(_UPLOAD, headers=self._hdr(), content=f.read())
            r1.raise_for_status()
            audio_url = r1.json()["upload_url"]

            body = {
                "audio_url": audio_url,
                "speech_models": [speech_model],  # API expects a list
                "speaker_labels": speaker_labels,
            }
            if language and language != "auto":
                body["language_code"] = language
            r2 = await client.post(_TRANSCRIPT, headers={**self._hdr(),
                                   "content-type": "application/json"}, json=body)
            if r2.status_code >= 400:
                raise RuntimeError(f"AssemblyAI {r2.status_code}: {r2.text[:300]}")
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

    # ── Streaming transcribe (Universal-Streaming v3) ───────────────────
    async def transcribe_stream(
        self, audio_path: str, config: dict
    ) -> AsyncIterator[Dict[str, Any]]:
        """Pipe a WAV through AssemblyAI's Universal-Streaming v3 endpoint
        and yield partials as Turn frames arrive.

        Wire shape per AssemblyAI docs:
          GET wss://streaming.assemblyai.com/v3/ws
            ?sample_rate=16000&encoding=pcm_s16le&token=<api_key>
          (api key may also live in Authorization header — query is simpler)

          server emits:
            {"type":"Begin", id, expires_at, ...}     (session-start)
            {"type":"Turn",  turn_order, transcript,
              end_of_turn:bool, turn_is_formatted:bool, words:[...]}
            {"type":"Termination", audio_duration_seconds, ...}

          client sends:
            <raw PCM s16le 16k mono bytes>
            {"type":"Terminate"}                       (end of session)
        """
        try:
            import websockets  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError("websockets pip pkg required") from e

        chunk_ms = int(config.get("chunk_ms", 100))

        # Read WAV → PCM s16le @ 16 kHz mono. AssemblyAI Universal-Streaming
        # only accepts 16 kHz mono int16; bail with a clear error if the
        # source clip doesn't match (the runner will surface this as
        # StageFailed).
        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            pcm = wf.readframes(wf.getnframes())

        if sampwidth != 2:
            raise RuntimeError(
                f"assemblyai streaming expects 16-bit PCM; got {sampwidth*8}-bit"
            )
        if sr != 16000:
            # Best-effort resample to 16 kHz. Falls back to scipy.signal.
            import numpy as np
            try:
                import scipy.signal  # type: ignore[import]
            except ImportError:
                raise RuntimeError(
                    f"assemblyai streaming needs 16 kHz audio; clip is {sr} Hz "
                    "and scipy isn't installed for resampling."
                )
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
                channels = 1
            samples = scipy.signal.resample_poly(samples, 16000, sr)
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            sr = 16000

        if channels > 1:
            import numpy as np
            samples = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels)
            pcm = samples.mean(axis=1).astype(np.int16).tobytes()
            channels = 1

        bytes_per_chunk = int(sr * (chunk_ms / 1000.0)) * 2  # 2 bytes/sample mono

        # AssemblyAI v3 requires speech_model on the WS handshake.
        # Valid values: 'universal-streaming-english',
        #               'universal-streaming-multilingual'.
        speech_model = config.get("streaming_model", "universal-streaming-english")
        url = (f"{_STREAMING_WS}?sample_rate={sr}"
               f"&speech_model={speech_model}"
               f"&format_turns=true")

        t0 = time.perf_counter()
        finalized_turns: List[str] = []           # list of formatted turn texts
        unfmt_partial = ""                        # current unformatted turn text
        all_words: List[Dict[str, Any]] = []

        async with websockets.connect(
            url,
            additional_headers={"Authorization": self._hdr()["authorization"]},
            max_size=None,
        ) as ws:

            async def _send():
                # Producer: pace audio in real time, then send Terminate.
                for i in range(0, len(pcm), bytes_per_chunk):
                    await ws.send(pcm[i : i + bytes_per_chunk])
                    await asyncio.sleep(chunk_ms / 1000.0)
                await ws.send(_json.dumps({"type": "Terminate"}))

            send_task = asyncio.create_task(_send())

            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        msg = _json.loads(raw)
                    except _json.JSONDecodeError:
                        continue

                    mtype = msg.get("type")
                    if mtype == "Begin":
                        continue   # session start ack
                    if mtype == "Termination":
                        break      # session end — drain done

                    if mtype != "Turn":
                        continue

                    transcript = msg.get("transcript", "") or ""
                    end_of_turn = bool(msg.get("end_of_turn", False))
                    is_formatted = bool(msg.get("turn_is_formatted", False))

                    if end_of_turn and is_formatted:
                        # Formatted final turn — promote into finalized list.
                        if transcript:
                            finalized_turns.append(transcript)
                        unfmt_partial = ""
                        # Capture word timing if present
                        for w in msg.get("words", []) or []:
                            all_words.append({
                                "word": w.get("text", ""),
                                "start": float(w.get("start", 0)) / 1000.0,
                                "end": float(w.get("end", 0)) / 1000.0,
                                "confidence": w.get("confidence"),
                                "speaker": None,
                            })
                    else:
                        # Unformatted partial within a turn.
                        unfmt_partial = transcript

                    accumulated = " ".join(finalized_turns)
                    if unfmt_partial:
                        accumulated = (accumulated + " " + unfmt_partial).strip()

                    yield {
                        "partial_text": accumulated,
                        "is_final": False,
                        "raw": msg,
                    }
            finally:
                send_task.cancel()
                try:
                    await send_task
                except (asyncio.CancelledError, Exception):
                    pass

        wall_s = time.perf_counter() - t0
        full_text = " ".join(finalized_turns).strip()
        duration_s = len(pcm) / (sr * 2)
        # Universal-Streaming pricing ~$0.15/hr → $0.0025/min
        cost_usd = round(duration_s / 60.0 * 0.0025, 6) if duration_s else None

        yield {
            "partial_text": full_text,
            "is_final": True,
            "text": full_text,
            "words": all_words,
            "language": "en",
            "duration_s": duration_s,
            "cost_usd": cost_usd,
            "wall_time_s": wall_s,
        }
