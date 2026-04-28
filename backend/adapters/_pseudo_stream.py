"""Chunked pseudo-streaming for batch ASR adapters.

Many ASR providers / self-host models only expose a batch ``transcribe()``
endpoint. The Playground UX is much nicer when the user sees text grow
as the audio is processed, so this helper adapts batch adapters into the
streaming-yielder shape the runner expects:

    async for ev in pseudo_stream_chunks(adapter, audio_path, config):
        # ev = {"partial_text": str, "is_final": bool, ...}
        ...

Strategy: split the input wav into ``chunk_seconds`` chunks (default 8 s),
call ``adapter.transcribe()`` on each chunk independently, and yield the
running cumulative text after every chunk. The final yield carries the
fully-merged result with ``is_final=True``.

Tradeoffs:
  - Cost multiplies for cloud APIs: a 32 s clip → 4 chunked calls vs 1
    batch. Set ``chunk_seconds`` higher to amortise.
  - Quality may drop slightly because cross-chunk context is lost (e.g.
    sentence-level punctuation, cross-chunk speaker diarization). For
    Playground "show me text as it streams" UX this is fine.
  - Word timestamps are offset by chunk start so the merged ``words``
    list times still line up with the original audio.

Used by every batch ASR adapter via a one-line ``transcribe_stream``
delegate so the user sees the same streaming feel from faster-whisper,
Gladia, Speechmatics, Groq, Parakeet, Canary-1B-flash, Canary-Qwen-2.5B.
"""
from __future__ import annotations

import os
import tempfile
import wave
from typing import Any, AsyncIterator, Dict


DEFAULT_CHUNK_SECONDS = 8.0
MIN_CHUNK_SECONDS = 0.5     # discard final tail shorter than this


async def pseudo_stream_chunks(
    adapter: Any,
    audio_path: str,
    config: dict,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
) -> AsyncIterator[Dict[str, Any]]:
    """Pseudo-stream a batch adapter by chunking the audio.

    Splits ``audio_path`` (16-bit PCM wav) into ``chunk_seconds`` chunks,
    calls ``adapter.transcribe()`` on each, and yields cumulative
    partials. The final yield carries the merged result with
    ``is_final=True`` so the runner promotes it to StageCompleted.

    Yields one ``is_final=True`` dict at the end no matter what — even
    if every chunk produced empty text — so the runner never sees the
    "streaming adapter ended without is_final yield" RuntimeError.
    """
    # Read whole wav so we can tell quickly if there's no point chunking.
    try:
        with wave.open(audio_path, "rb") as wf:
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            ch = wf.getnchannels()
            n_frames = wf.getnframes()
            all_pcm = wf.readframes(n_frames)
    except Exception:
        # Fall back to a straight-up batch call if we can't parse the wav
        # (e.g. mp3, opus, vorbis — the adapter may handle it natively).
        result = await adapter.transcribe(audio_path, config)
        text = result.get("text", "")
        yield {
            "partial_text": text,
            "is_final": True,
            **{k: v for k, v in result.items() if k not in ("partial_text", "is_final")},
        }
        return

    duration_s = n_frames / sr if sr else 0.0
    bytes_per_frame = sw * ch
    chunk_frames = max(1, int(chunk_seconds * sr))
    chunk_bytes = chunk_frames * bytes_per_frame

    # Short clip: skip chunking, single batch call. Still yield two events
    # (one mid-stream, one final) so the runner's stream-shaped contract
    # is satisfied and the UI gets to flash a partial cursor briefly.
    if duration_s <= chunk_seconds + 0.5:
        result = await adapter.transcribe(audio_path, config)
        text = (result.get("text") or "").strip()
        yield {"partial_text": text, "is_final": False}
        yield {
            "partial_text": text,
            "is_final": True,
            **{k: v for k, v in result.items() if k not in ("partial_text", "is_final")},
        }
        return

    # Multi-chunk path.
    accumulated_text = ""
    accumulated_words: list = []
    accumulated_cost = 0.0
    last_result: Dict[str, Any] = {}
    time_offset = 0.0   # seconds into the original clip
    min_chunk_bytes = int(MIN_CHUNK_SECONDS * sr) * bytes_per_frame

    for offset in range(0, len(all_pcm), chunk_bytes):
        chunk_pcm = all_pcm[offset : offset + chunk_bytes]
        if len(chunk_pcm) < min_chunk_bytes:
            break

        # Write a temp wav for this chunk so the adapter's transcribe()
        # can read it just like the full clip. soundfile/ffmpeg-friendly
        # 16-bit PCM regardless of the input encoding.
        tmp_path: str = ""
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_path = tmp.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(ch)
                wf.setsampwidth(sw)
                wf.setframerate(sr)
                wf.writeframes(chunk_pcm)

            chunk_result = await adapter.transcribe(tmp_path, config)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        chunk_text = (chunk_result.get("text") or "").strip()
        if chunk_text:
            accumulated_text = (
                accumulated_text + " " + chunk_text
            ).strip() if accumulated_text else chunk_text

        # Offset word timestamps so they reference the original clip.
        for w in (chunk_result.get("words") or []):
            try:
                accumulated_words.append({
                    **w,
                    "start": float(w.get("start", 0.0)) + time_offset,
                    "end": float(w.get("end", 0.0)) + time_offset,
                })
            except Exception:
                accumulated_words.append(w)

        accumulated_cost += float(chunk_result.get("cost_usd") or 0)
        last_result = chunk_result
        time_offset += len(chunk_pcm) / bytes_per_frame / sr

        yield {
            "partial_text": accumulated_text,
            "is_final": False,
        }

    # Final yield — runner promotes this to StageCompleted.
    final: Dict[str, Any] = {
        "partial_text": accumulated_text,
        "is_final": True,
        "text": accumulated_text,
        "words": accumulated_words,
        "language": last_result.get("language", config.get("language", "en")),
        "duration_s": duration_s,
        "cost_usd": accumulated_cost or last_result.get("cost_usd"),
        "wall_time_s": last_result.get("wall_time_s"),
    }
    # Carry through any vendor-specific keys the last chunk had (e.g.
    # raw_response from the final chunk only — not perfect, but enough
    # for debugging).
    for k, v in last_result.items():
        if k not in final and k not in ("partial_text", "is_final", "text", "words"):
            final[k] = v
    yield final
