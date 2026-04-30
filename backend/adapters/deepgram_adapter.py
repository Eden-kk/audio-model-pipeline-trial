"""Deepgram Nova-3 ASR adapter — supports both batch and streaming.

Batch  : POST /v1/listen with the full WAV → returns the final transcript.
Stream : WSS connection that emits Results frames as the audio plays through;
         each frame may be a partial (is_final=false) or finalised utterance.
         The adapter pipes a pre-recorded WAV through the streaming endpoint
         in real-time-paced chunks so the user sees the same UX as live mic
         streaming.

Env: DEEPGRAM_API_KEY
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

_ENDPOINT = "https://api.deepgram.com/v1/listen"
_WS_ENDPOINT = "wss://api.deepgram.com/v1/listen"

_DEFAULT_PARAMS = {
    "model": "nova-3",
    "smart_format": "true",
    "punctuate": "true",
    "diarize": "true",
    "utterances": "true",
}


class DeepgramAdapter:
    # ── Adapter identity ────────────────────────────────────────────────
    id = "deepgram"
    category = "asr"
    display_name = "Deepgram Nova-3"
    hosting = "cloud"
    vendor = "Deepgram"
    is_streaming = True   # transcribe_stream() implemented via WSS
    supported_languages: List[str] = ["auto", "en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "hi", "ar", "ru"]

    inputs: List[Dict[str, str]] = [
        {"name": "audio", "type": "audio_file"},
    ]
    outputs: List[Dict[str, str]] = [
        {"name": "text", "type": "text"},
        {"name": "words", "type": "word_timing"},
    ]

    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "default": "en",
                "description": "BCP-47 language code or 'auto' for auto-detect.",
            },
            "model": {
                "type": "string",
                "default": "nova-3",
                "description": "Deepgram model variant.",
            },
            "diarize": {
                "type": "boolean",
                "default": True,
                "description": "Enable speaker diarization.",
            },
        },
    }
    cost_per_call_estimate_usd: Optional[float] = None  # computed per call from duration

    # ── Construction ────────────────────────────────────────────────────
    def __init__(self) -> None:
        # Key is read lazily at call time so the adapter can be registered at
        # import time even when the env var is absent (e.g. during tests).
        pass

    def _api_key(self) -> str:
        key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is not set. "
                "Add it to backend/.env before calling Deepgram."
            )
        return key

    # ── Core HTTP call ──────────────────────────────────────────────────
    async def _call(
        self,
        audio_path: str,
        *,
        language: str,
        model: str,
        diarize: bool,
    ) -> dict:
        lang_param = language if language != "auto" else "multi"
        params = {
            **_DEFAULT_PARAMS,
            "model": model,
            "language": lang_param,
            "diarize": "true" if diarize else "false",
            "detect_language": "true" if lang_param == "multi" else "false",
        }
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                _ENDPOINT,
                params=params,
                headers={
                    "Authorization": f"Token {self._api_key()}",
                    "Content-Type": "audio/wav",
                },
                content=audio_bytes,
            )
        resp.raise_for_status()
        return resp.json()

    # ── Public transcribe ───────────────────────────────────────────────
    async def transcribe(self, audio_path: str, config: dict) -> dict:
        """Run ASR via Deepgram Nova-3 and return a normalised result dict.

        Returns:
            {text, words, language, duration_s, cost_usd, raw_response}
        """
        language = config.get("language", "en")
        model = config.get("model", "nova-3")
        diarize = bool(config.get("diarize", True))

        t0 = time.perf_counter()
        body = await self._call(
            audio_path, language=language, model=model, diarize=diarize
        )
        wall_s = time.perf_counter() - t0

        results = body.get("results", {})
        channels = results.get("channels", [])
        if not channels:
            raise ValueError("Deepgram returned no channels in response")

        alt = channels[0]["alternatives"][0]
        text = alt.get("transcript", "")
        detected_language = channels[0].get("detected_language", language)

        words = []
        for w in alt.get("words", []) or []:
            words.append({
                "word": w.get("word", ""),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "confidence": w.get("confidence"),
                "speaker": str(w["speaker"]) if "speaker" in w else None,
            })

        duration_s = float(
            results.get("metadata", {}).get("duration")
            or body.get("metadata", {}).get("duration", 0)
        )
        # Nova-3 batch pricing: ~$0.0043 / minute
        cost_usd = round(duration_s / 60.0 * 0.0043, 6) if duration_s else None

        return {
            "text": text,
            "words": words,
            "language": detected_language,
            "duration_s": duration_s,
            "cost_usd": cost_usd,
            "wall_time_s": wall_s,
            "raw_response": body,
        }

    # ── Streaming transcribe ────────────────────────────────────────────
    async def transcribe_stream(
        self, audio_path: str, config: dict
    ) -> AsyncIterator[Dict[str, Any]]:
        """Open a Deepgram WSS connection, pipe the audio file through in
        ~real-time-paced chunks, and yield partial transcripts as they
        arrive. Final yield carries the full text + words.

        WSS frame shapes Deepgram emits (we only read 'Results' frames):
          {type: "Results", channel: {alternatives: [{transcript, words}]},
           is_final: bool, speech_final: bool, ...}
          {type: "Metadata", ...}            (sent at session start/end)
          {type: "UtteranceEnd", ...}        (between utterances)
        """
        try:
            import websockets  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError(
                "websockets pip pkg not installed; add it to requirements.txt"
            ) from e
        import wave

        language = config.get("language", "en")
        model = config.get("model", "nova-3")
        diarize = bool(config.get("diarize", True))
        # Pacing: real-time-ish → send 100 ms of audio every 100 ms.
        # Faster pacing finishes sooner but costs the live-stream feel.
        chunk_ms = int(config.get("chunk_ms", 100))

        # Read PCM samples from the WAV. Deepgram's WSS accepts raw PCM
        # only when we tell it what we're sending (sample rate + channels +
        # encoding); we set those via query params.
        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            pcm = wf.readframes(wf.getnframes())

        if sampwidth != 2:
            raise RuntimeError(
                f"deepgram streaming expects 16-bit PCM; got {sampwidth*8}-bit"
            )

        bytes_per_chunk = int(sr * (chunk_ms / 1000.0)) * channels * sampwidth

        params = {
            "model": model,
            "language": language if language != "auto" else "multi",
            "encoding": "linear16",
            "sample_rate": str(sr),
            "channels": str(channels),
            "smart_format": "true",
            "punctuate": "true",
            "interim_results": "true",   # critical: enables partials
            "diarize": "true" if diarize else "false",
        }
        url = _WS_ENDPOINT + "?" + "&".join(f"{k}={v}" for k, v in params.items())

        t0 = time.perf_counter()
        full_words: List[Dict[str, Any]] = []
        latest_partial = ""
        final_segments: List[str] = []
        detected_language = language

        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Token {self._api_key()}"},
            max_size=None,
        ) as ws:
            # Producer: pace the audio
            async def _send():
                for i in range(0, len(pcm), bytes_per_chunk):
                    await ws.send(pcm[i : i + bytes_per_chunk])
                    await asyncio.sleep(chunk_ms / 1000.0)
                # Tell Deepgram we're done so it flushes final results.
                await ws.send(_json.dumps({"type": "CloseStream"}))

            send_task = asyncio.create_task(_send())

            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        msg = _json.loads(raw)
                    except _json.JSONDecodeError:
                        continue

                    if msg.get("type") != "Results":
                        continue

                    alts = msg.get("channel", {}).get("alternatives", [])
                    if not alts:
                        continue
                    alt = alts[0]
                    text = alt.get("transcript", "") or ""
                    is_final = bool(msg.get("is_final"))

                    if is_final and text:
                        final_segments.append(text)
                        latest_partial = ""
                        for w in alt.get("words", []) or []:
                            full_words.append({
                                "word": w.get("word", ""),
                                "start": float(w.get("start", 0.0)),
                                "end": float(w.get("end", 0.0)),
                                "confidence": w.get("confidence"),
                                "speaker": (str(w["speaker"])
                                            if "speaker" in w else None),
                            })
                    else:
                        latest_partial = text

                    accumulated = " ".join(final_segments)
                    if latest_partial:
                        accumulated = (
                            accumulated + " " + latest_partial
                        ).strip()

                    yield {
                        "partial_text": accumulated,
                        "is_final": False,   # mid-stream — runner will fire stage.progress
                        "raw": msg,
                    }

                    if msg.get("type") == "Results" and msg.get("from_finalize"):
                        break
            finally:
                send_task.cancel()
                try:
                    await send_task
                except (asyncio.CancelledError, Exception):
                    pass

        wall_s = time.perf_counter() - t0
        full_text = " ".join(final_segments).strip()
        duration_s = len(pcm) / (sr * channels * sampwidth)
        cost_usd = round(duration_s / 60.0 * 0.0059, 6) if duration_s else None  # streaming pricing

        # Final yield — runner promotes this to stage.finished.
        yield {
            "partial_text": full_text,
            "is_final": True,
            "text": full_text,
            "words": full_words,
            "language": detected_language,
            "duration_s": duration_s,
            "cost_usd": cost_usd,
            "wall_time_s": wall_s,
        }
