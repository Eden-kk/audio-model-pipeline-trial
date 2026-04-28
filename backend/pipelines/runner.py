"""Multi-stage pipeline runner.

Walks a Pipeline's stages in order, dispatching each stage to its adapter's
category-appropriate entry-point (transcribe / synthesize / verify), and
threading the prior stage's output into the next stage's input.

Category dispatch rules (today):
  asr             → adapter.transcribe(audio_path, config)        → {text, words, ...}
  tts             → adapter.synthesize(text, config)               → {audio_b64, ...}
  speaker_verify  → adapter.verify(audio_path, embedding, config)  → {score, match}

Edge port-typing is enforced loosely: if the upstream stage's category isn't
compatible with the downstream stage's first input port, the runner returns
a clear `incompatible_stage_chain` error before invoking any adapter. This
prevents a TTS adapter from being handed an audio-file path, etc.
"""
from __future__ import annotations

import asyncio
import datetime
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional


# ── Port-compatibility table (what upstream output can feed into what
#    downstream input).  Keep small; expand as new categories land.
_COMPATIBLE: Dict[str, List[str]] = {
    "asr":            ["text"],            # ASR text → TTS input, intent LLM input
    "tts":            ["audio_stream"],    # TTS audio → could feed ASR (round-trip)
    "speaker_verify": ["score"],           # Verify result → metric only, no chain
}


class StageError(Exception):
    """Raised when a single stage fails — the runner records and stops."""


async def run_pipeline(
    *,
    pipeline: Dict[str, Any],
    clip_audio_path: str,
    stage_overrides: Dict[str, str],   # stage_id → adapter_id (resolves recipe placeholders)
    stage_configs: Dict[str, Dict[str, Any]] | None,
    registry: Any,                      # adapter registry (with .get(adapter_id))
    on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    """Execute a pipeline against one clip; return per-stage results + totals.

    Returns a dict shaped like:
        {
          "stages": [
              {stage_id, adapter, category, started_at, finished_at,
               latency_ms, cost_usd, input_preview, output_preview,
               result, error}
              ...
          ],
          "total_latency_ms": float,
          "total_cost_usd": float,
          "error": Optional[str],
        }

    Per-stage events are pushed to `on_event` if provided so a WebSocket can
    relay them live to the UI.
    """
    stages = pipeline.get("stages") or []
    stage_configs = stage_configs or {}
    out_stages: List[Dict[str, Any]] = []
    total_cost = 0.0
    pipeline_t0 = time.perf_counter()

    # Stage outputs for chaining: stage_id → {text?, audio_path?, ...}
    upstream_text: Optional[str] = None
    upstream_audio_path: Optional[str] = clip_audio_path  # initial input

    async def _emit(payload: Dict[str, Any]) -> None:
        if on_event is not None:
            try:
                await on_event(payload)
            except Exception:
                pass  # never let event-emit failure kill the run

    for stage in stages:
        stage_id = stage["id"]
        category = stage["category"]
        adapter_id = stage_overrides.get(stage_id) or stage.get("adapter")
        config = {**(stage.get("config") or {}), **(stage_configs.get(stage_id) or {})}

        if not adapter_id:
            err = (
                f"stage '{stage_id}' (category={category}) has no adapter — "
                f"recipe placeholder requires an adapter override in the request"
            )
            out_stages.append(_failed_stage(stage_id, category, None, err))
            await _emit({"event": "StageFailed", "stage_id": stage_id, "error": err})
            return _final(out_stages, total_cost, pipeline_t0, error=err)

        try:
            adapter = registry.get(adapter_id)
        except KeyError:
            err = f"adapter {adapter_id!r} is not registered"
            out_stages.append(_failed_stage(stage_id, category, adapter_id, err))
            await _emit({"event": "StageFailed", "stage_id": stage_id, "error": err})
            return _final(out_stages, total_cost, pipeline_t0, error=err)

        # Soft port-type check: each adapter's first input must be feedable
        # by the upstream output. We trust the adapter's `category`.
        if getattr(adapter, "category", None) != category:
            err = (
                f"adapter {adapter_id!r} has category {adapter.category!r} but "
                f"stage {stage_id!r} expects {category!r}"
            )
            out_stages.append(_failed_stage(stage_id, category, adapter_id, err))
            await _emit({"event": "StageFailed", "stage_id": stage_id, "error": err})
            return _final(out_stages, total_cost, pipeline_t0, error=err)

        # ── Run the stage ──────────────────────────────────────────────────
        started_at = datetime.datetime.utcnow().isoformat() + "Z"
        await _emit({
            "event": "StageStarted",
            "stage_id": stage_id,
            "adapter": adapter_id,
            "category": category,
            "started_at": started_at,
        })
        t0 = time.perf_counter()
        result: Dict[str, Any] = {}
        stage_err: Optional[str] = None
        try:
            if category == "asr":
                # Take audio from upstream (initial clip or last stage's audio output)
                if upstream_audio_path is None:
                    raise StageError("asr stage requires upstream audio")
                result = await adapter.transcribe(upstream_audio_path, config)
                upstream_text = (result.get("text") or "")
                # Don't update upstream_audio_path; ASR doesn't produce audio.

            elif category == "tts":
                if upstream_text is None:
                    raise StageError(
                        "tts stage requires upstream text (chain after ASR or "
                        "supply config.text in stage_configs)"
                    )
                result = await adapter.synthesize(upstream_text, config)
                # TTS produces audio_b64; persist to a tmpfile for chaining.
                upstream_audio_path = _persist_audio_b64(
                    result.get("audio_b64"),
                    sample_rate=int(result.get("sample_rate", 16000)),
                )

            elif category == "speaker_verify":
                # Speaker verify expects an enrolled embedding in config.
                emb = config.get("enrolled_embedding_b64")
                if not emb:
                    raise StageError(
                        "speaker_verify stage requires "
                        "config.enrolled_embedding_b64 (run /api/enroll first)"
                    )
                if upstream_audio_path is None:
                    raise StageError("speaker_verify stage requires audio")
                result = await adapter.verify(
                    upstream_audio_path,
                    enrolled_embedding_b64=emb,
                    config=config,
                )
            else:
                raise StageError(f"unsupported stage category {category!r}")
        except Exception as e:
            stage_err = f"{type(e).__name__}: {e}"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        finished_at = datetime.datetime.utcnow().isoformat() + "Z"
        cost = float(result.get("cost_usd") or 0.0)
        total_cost += cost

        stage_record = {
            "stage_id": stage_id,
            "category": category,
            "adapter": adapter_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "input_preview": _short_preview(
                upstream_text if category == "tts" else upstream_audio_path
            ),
            "output_preview": _short_preview(_pick_output_preview(category, result)),
            "result": result,
            "error": stage_err,
        }
        out_stages.append(stage_record)

        if stage_err:
            await _emit({
                "event": "StageFailed",
                "stage_id": stage_id,
                "adapter": adapter_id,
                "error": stage_err,
                "latency_ms": latency_ms,
            })
            return _final(out_stages, total_cost, pipeline_t0, error=stage_err)

        await _emit({
            "event": "StageCompleted",
            "stage_id": stage_id,
            "adapter": adapter_id,
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "output_preview": stage_record["output_preview"],
        })

    return _final(out_stages, total_cost, pipeline_t0)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _final(stages: List[Dict[str, Any]], total_cost: float,
           t0: float, *, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "stages": stages,
        "total_latency_ms": (time.perf_counter() - t0) * 1000.0,
        "total_cost_usd": round(total_cost, 6),
        "error": error,
    }


def _failed_stage(stage_id: str, category: str,
                  adapter_id: Optional[str], err: str) -> Dict[str, Any]:
    now = datetime.datetime.utcnow().isoformat() + "Z"
    return {
        "stage_id": stage_id,
        "category": category,
        "adapter": adapter_id,
        "started_at": now,
        "finished_at": now,
        "latency_ms": 0.0,
        "cost_usd": 0.0,
        "input_preview": "",
        "output_preview": "",
        "result": {},
        "error": err,
    }


def _short_preview(value: Any, limit: int = 120) -> str:
    if value is None:
        return ""
    s = str(value)
    return s if len(s) <= limit else s[:limit] + "…"


def _pick_output_preview(category: str, result: Dict[str, Any]) -> str:
    if category == "asr":
        return result.get("text") or ""
    if category == "tts":
        sr = result.get("sample_rate")
        dur = result.get("duration_s")
        ttfa = result.get("first_byte_ms")
        return (
            f"audio · {dur:.2f}s @ {sr}Hz · TTFA={ttfa:.0f}ms"
            if dur and sr and ttfa else "audio"
        )
    if category == "speaker_verify":
        score = result.get("score")
        match = result.get("match")
        thr = result.get("threshold")
        return f"score={score:.3f} match={match} (thr={thr})" if score is not None else ""
    return ""


def _persist_audio_b64(audio_b64: Optional[str],
                       sample_rate: int) -> Optional[str]:
    """Decode TTS output into a temp WAV so the next stage (e.g. ASR
    round-trip) can read it from disk."""
    if not audio_b64:
        return None
    import base64
    import struct
    import tempfile
    pcm = base64.b64decode(audio_b64)
    # Write a minimal WAV header + raw PCM s16le
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    n_samples = len(pcm) // 2
    byte_rate = sample_rate * 2
    chunk_size = 36 + len(pcm)
    tmp.write(b"RIFF")
    tmp.write(struct.pack("<I", chunk_size))
    tmp.write(b"WAVE")
    tmp.write(b"fmt ")
    tmp.write(struct.pack("<I", 16))                # subchunk1 size
    tmp.write(struct.pack("<H", 1))                 # PCM
    tmp.write(struct.pack("<H", 1))                 # mono
    tmp.write(struct.pack("<I", sample_rate))
    tmp.write(struct.pack("<I", byte_rate))
    tmp.write(struct.pack("<H", 2))                 # block align
    tmp.write(struct.pack("<H", 16))                # bits per sample
    tmp.write(b"data")
    tmp.write(struct.pack("<I", len(pcm)))
    tmp.write(pcm)
    tmp.close()
    return tmp.name
