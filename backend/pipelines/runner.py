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
import json as _json
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

log = logging.getLogger("trial-app.runner")


def _load_enrollment_record(profile_id: str = "wearer") -> Optional[Dict[str, Any]]:
    """Return the full enrollment dict (embedding_b64, adapter, embedding_dim,
    enrolled_at, …) saved by POST /api/enroll, or None if missing.

    Multi-profile (Plan A familiar voices) reads different profile_ids; v1
    uses 'wearer' by default. Returning the whole record lets the runner
    cross-check that the speaker_verify stage's adapter matches the one the
    embedding was extracted with — Resemblyzer (256-d) vs pyannote_verify
    (512-d) embeddings live in different vector spaces, so cosine-comparing
    them throws a numpy shape error.
    """
    base = Path(os.environ.get("DATA_DIR", "data"))
    p = base / "enrollments" / f"{profile_id}.json"
    if not p.exists():
        return None
    try:
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_enrolled_embedding(profile_id: str = "wearer") -> Optional[str]:
    """Backwards-compatible shim — returns just embedding_b64."""
    rec = _load_enrollment_record(profile_id)
    return rec.get("embedding_b64") if rec else None


# ── Port-compatibility table (what upstream output can feed into what
#    downstream input).  Keep small; expand as new categories land.
_COMPATIBLE: Dict[str, List[str]] = {
    "asr":            ["text"],            # ASR text → TTS input, intent LLM input
    "tts":            ["audio_stream"],    # TTS audio → could feed ASR (round-trip)
    "speaker_verify": ["score"],           # Verify result → metric only, no chain
}


class StageError(Exception):
    """Raised when a single stage fails — the runner records and stops."""


def _topological_waves(
    stages: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Group stages into parallel-executable waves via Kahn-style layering.

    A "wave" is the set of stages whose upstream dependencies (per `edges`)
    have all been placed in some earlier wave. Stages within a wave have no
    edges between them and can run concurrently; we still preserve their
    declaration order for deterministic UI layout.

    Returns: list of waves, each wave is a list of stage dicts.
    """
    by_id = {s["id"]: s for s in stages}
    deps: Dict[str, set] = {
        s["id"]: {e["from"] for e in edges if e["to"] == s["id"] and e["from"] in by_id}
        for s in stages
    }
    placed: set = set()
    waves: List[List[Dict[str, Any]]] = []
    # Walk in declared order so layout columns mirror the recipe author's intent
    while len(placed) < len(stages):
        wave = [
            s for s in stages
            if s["id"] not in placed and deps[s["id"]] <= placed
        ]
        if not wave:
            # Cycle (or unreachable dep) — degrade to running the remaining stages
            # one at a time so we still make progress instead of looping forever.
            remaining = [s for s in stages if s["id"] not in placed]
            wave = remaining[:1]
        waves.append(wave)
        placed |= {s["id"] for s in wave}
    return waves


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
    edges = pipeline.get("edges") or []
    stage_configs = stage_configs or {}
    out_stages: List[Dict[str, Any]] = []
    total_cost = 0.0
    pipeline_t0 = time.perf_counter()

    # Shared upstream state — mutated *between* waves only, so each stage
    # within a wave sees an immutable snapshot. Stages with no edge between
    # them have no real data dependency, so this is sound; the snapshot is
    # taken at wave-start and merged after asyncio.gather() finishes.
    upstream_text: Optional[str] = None
    upstream_audio_path: Optional[str] = clip_audio_path  # initial input
    upstream_words: List[Dict[str, Any]] = []
    upstream_language: Optional[str] = None
    upstream_speaker_segments: List[Dict[str, Any]] = []
    # All stage results keyed by stage_id so terminal stages (dispatch) can
    # reach back to whatever upstream stage produced the envelope.
    stage_outputs: Dict[str, Dict[str, Any]] = {}

    async def _emit(payload: Dict[str, Any]) -> None:
        if on_event is not None:
            try:
                await on_event(payload)
            except Exception:
                pass  # never let event-emit failure kill the run

    async def _run_one_stage(
        stage: Dict[str, Any],
        wave_idx: int,
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single stage against an immutable snapshot of upstream
        state. Returns a stage_record dict augmented with a `_delta` key
        (popped before append) holding the upstream-state changes the caller
        should merge after the wave finishes.
        """
        stage_id = stage["id"]
        category = stage["category"]
        adapter_id = stage_overrides.get(stage_id) or stage.get("adapter")
        config = {**(stage.get("config") or {}), **(stage_configs.get(stage_id) or {})}

        if not adapter_id:
            err = (
                f"stage '{stage_id}' (category={category}) has no adapter — "
                f"recipe placeholder requires an adapter override in the request"
            )
            rec = _failed_stage(stage_id, category, None, err)
            rec["wave"] = wave_idx
            rec["_delta"] = {}
            await _emit({"event": "StageFailed", "stage_id": stage_id, "error": err})
            return rec

        try:
            adapter = registry.get(adapter_id)
        except KeyError:
            err = f"adapter {adapter_id!r} is not registered"
            rec = _failed_stage(stage_id, category, adapter_id, err)
            rec["wave"] = wave_idx
            rec["_delta"] = {}
            await _emit({"event": "StageFailed", "stage_id": stage_id, "error": err})
            return rec

        if getattr(adapter, "category", None) != category:
            err = (
                f"adapter {adapter_id!r} has category {adapter.category!r} but "
                f"stage {stage_id!r} expects {category!r}"
            )
            rec = _failed_stage(stage_id, category, adapter_id, err)
            rec["wave"] = wave_idx
            rec["_delta"] = {}
            await _emit({"event": "StageFailed", "stage_id": stage_id, "error": err})
            return rec

        snap_text = snapshot.get("text")
        snap_audio = snapshot.get("audio_path")
        snap_words = snapshot.get("words") or []
        snap_language = snapshot.get("language")
        snap_segments = snapshot.get("speaker_segments") or []
        snap_outputs = snapshot.get("stage_outputs") or {}

        started_at = datetime.datetime.utcnow().isoformat() + "Z"
        log.info(
            f"stage start  | wave={wave_idx} id={stage_id} "
            f"category={category} adapter={adapter_id}"
        )
        await _emit({
            "event": "StageStarted",
            "stage_id": stage_id,
            "adapter": adapter_id,
            "category": category,
            "wave": wave_idx,
            "started_at": started_at,
        })

        t0 = time.perf_counter()
        result: Dict[str, Any] = {}
        stage_err: Optional[str] = None
        delta: Dict[str, Any] = {}
        try:
            if category == "asr":
                if snap_audio is None:
                    raise StageError("asr stage requires upstream audio")
                merged_config = {**config}
                if snap_language and "language" not in config:
                    merged_config["language"] = snap_language
                result = await adapter.transcribe(snap_audio, merged_config)
                delta["text"] = result.get("text") or ""
                delta["words"] = result.get("words") or []
                if result.get("language"):
                    delta["language"] = result.get("language")

            elif category == "tts":
                if snap_text is None:
                    raise StageError(
                        "tts stage requires upstream text (chain after ASR or "
                        "supply config.text in stage_configs)"
                    )
                result = await adapter.synthesize(snap_text, config)
                delta["audio_path"] = _persist_audio_b64(
                    result.get("audio_b64"),
                    sample_rate=int(result.get("sample_rate", 16000)),
                )

            elif category == "lid":
                if snap_audio is None:
                    raise StageError("lid stage requires audio")
                result = await adapter.lid(snap_audio, config)
                delta["language"] = result.get("language")

            elif category == "speaker_verify":
                emb = config.get("enrolled_embedding_b64")
                if not emb:
                    profile_id = config.get("profile_id", "wearer")
                    rec = _load_enrollment_record(profile_id)
                    if not rec or not rec.get("embedding_b64"):
                        raise StageError(
                            f"speaker_verify stage needs an enrolled embedding. "
                            f"Either pass config.enrolled_embedding_b64 inline, "
                            f"or POST /api/enroll first to save profile "
                            f"'{profile_id}' to disk (no enrollment file at "
                            f"data/enrollments/{profile_id}.json)."
                        )
                    # Adapter cross-check: pyannote_verify produces 512-d
                    # ECAPA embeddings, Resemblyzer produces 256-d d-vectors,
                    # and the two are NOT comparable (different vector
                    # spaces). If we let a mismatched call through, numpy
                    # throws a cryptic shape error mid-pipeline; surface it
                    # here with a clean fix-it message instead.
                    enrolled_adapter = rec.get("adapter")
                    if enrolled_adapter and enrolled_adapter != adapter_id:
                        raise StageError(
                            f"speaker_verify adapter mismatch — profile "
                            f"'{profile_id}' was enrolled with "
                            f"{enrolled_adapter!r} ({rec.get('embedding_dim')}-d), "
                            f"but this stage is set to {adapter_id!r}. "
                            f"Either (a) switch the speaker_tag stage's adapter "
                            f"back to {enrolled_adapter!r}, or (b) re-enroll "
                            f"the wearer with {adapter_id!r} via Settings → "
                            f"Wearer enrollment."
                        )
                    emb = rec["embedding_b64"]
                if snap_audio is None:
                    raise StageError("speaker_verify stage requires audio")
                mode = config.get("mode", "overall")
                if mode == "segments":
                    result = await adapter.verify_segments(
                        snap_audio,
                        enrolled_embedding_b64=emb,
                        config=config,
                    )
                    delta["speaker_segments"] = result.get("segments") or []
                else:
                    result = await adapter.verify(
                        snap_audio,
                        enrolled_embedding_b64=emb,
                        config=config,
                    )

            elif category == "intent_llm":
                if snap_text is None:
                    raise StageError(
                        "intent_llm stage requires an upstream ASR transcript"
                    )
                payload = {
                    "text": snap_text,
                    "words": snap_words,
                    "speaker_segments": snap_segments,
                    "language": snap_language or "en",
                }
                result = await adapter.infer(payload, config)

            elif category == "dispatch":
                envelope: Dict[str, Any]
                if "envelope" in config:
                    envelope = config["envelope"]
                else:
                    envelope = {}
                    for s_id in reversed(list(snap_outputs.keys())):
                        prev = snap_outputs[s_id]
                        if any(k in prev for k in ("memory_doc", "tool_calls", "salient_facts")):
                            envelope = {k: prev.get(k)
                                        for k in ("memory_doc", "tool_calls", "salient_facts")
                                        if k in prev}
                            break
                if not envelope:
                    raise StageError(
                        "dispatch stage requires upstream envelope "
                        "(intent_llm stage with memory_doc/tool_calls/salient_facts)"
                    )
                result = await adapter.dispatch(envelope, config)

            else:
                raise StageError(f"unsupported stage category {category!r}")
        except Exception as e:
            stage_err = f"{type(e).__name__}: {e}"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        finished_at = datetime.datetime.utcnow().isoformat() + "Z"
        cost = float(result.get("cost_usd") or 0.0)

        stage_record = {
            "stage_id": stage_id,
            "category": category,
            "adapter": adapter_id,
            "wave": wave_idx,
            "started_at": started_at,
            "finished_at": finished_at,
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "input_preview": _short_preview(
                snap_text if category == "tts" else snap_audio
            ),
            "output_preview": _short_preview(_pick_output_preview(category, result)),
            "result": result,
            "error": stage_err,
            "_delta": delta,
        }

        if stage_err:
            log.warning(
                f"stage FAIL   | wave={wave_idx} id={stage_id} "
                f"latency={latency_ms:.0f}ms err={stage_err}"
            )
            await _emit({
                "event": "StageFailed",
                "stage_id": stage_id,
                "adapter": adapter_id,
                "error": stage_err,
                "latency_ms": latency_ms,
            })
        else:
            log.info(
                f"stage done   | wave={wave_idx} id={stage_id} "
                f"latency={latency_ms:.0f}ms cost=${cost:.5f} "
                f"preview={stage_record['output_preview'][:80]!r}"
            )
            await _emit({
                "event": "StageCompleted",
                "stage_id": stage_id,
                "adapter": adapter_id,
                "wave": wave_idx,
                "latency_ms": latency_ms,
                "cost_usd": cost,
                "output_preview": stage_record["output_preview"],
            })

        return stage_record

    waves = _topological_waves(stages, edges)
    if waves:
        log.info(
            "pipeline waves | "
            + " | ".join(
                f"w{i}=[{','.join(s['id'] for s in w)}]"
                for i, w in enumerate(waves)
            )
        )

    for wave_idx, wave in enumerate(waves):
        snapshot = {
            "text": upstream_text,
            "audio_path": upstream_audio_path,
            "words": upstream_words,
            "language": upstream_language,
            "speaker_segments": upstream_speaker_segments,
            "stage_outputs": dict(stage_outputs),
        }
        # Run every stage in the wave concurrently against the same snapshot.
        records = await asyncio.gather(
            *[_run_one_stage(s, wave_idx, snapshot) for s in wave]
        )

        # Merge state deltas in declaration order for determinism. Stages in
        # the same wave that both write the same field (rare; means recipe
        # author put two ASRs side-by-side) get last-writer-wins by recipe
        # order, which is the most predictable rule.
        for rec in records:
            delta = rec.pop("_delta", {})
            stage_outputs[rec["stage_id"]] = rec["result"]
            if "text" in delta:
                upstream_text = delta["text"]
            if "audio_path" in delta:
                upstream_audio_path = delta["audio_path"]
            if "words" in delta:
                upstream_words = delta["words"]
            if "language" in delta and delta["language"]:
                upstream_language = delta["language"]
            if "speaker_segments" in delta:
                upstream_speaker_segments = delta["speaker_segments"]
            total_cost += rec["cost_usd"]
            out_stages.append(rec)

        # If any stage in this wave failed, stop — don't run downstream waves
        # whose inputs would be missing.
        first_err = next((r["error"] for r in records if r["error"]), None)
        if first_err:
            return _final(out_stages, total_cost, pipeline_t0, error=first_err)

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
        # verify_segments() returns {segments: [...], n_segments, ...}
        if "n_segments" in result:
            user_n = sum(1 for s in result.get("segments") or [] if s.get("is_user"))
            n = result["n_segments"]
            return f"{n} segments · {user_n} user · {n - user_n} other"
        score = result.get("score")
        match = result.get("match")
        thr = result.get("threshold")
        return f"score={score:.3f} match={match} (thr={thr})" if score is not None else ""
    if category == "lid":
        lang = result.get("language")
        conf = result.get("confidence")
        return f"lang={lang} ({conf:.2f})" if lang and conf is not None else f"lang={lang}"
    if category == "intent_llm":
        n_calls = len(result.get("tool_calls") or [])
        n_facts = len(result.get("salient_facts") or [])
        memory_chars = len(result.get("memory_doc") or "")
        return f"{n_calls} tool_calls · {n_facts} facts · memory_doc {memory_chars}c"
    if category == "dispatch":
        ack = result.get("ack") or ""
        bytes_w = result.get("bytes_written")
        return f"{ack}" + (f" ({bytes_w}b)" if bytes_w else "")
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
