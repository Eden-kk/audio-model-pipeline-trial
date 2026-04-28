"""FastAPI entrypoint for the audio-trial backend — Slice 0.

Routes
------
GET  /api/health                   — liveness probe
GET  /api/adapters                 — list registered adapters
POST /api/clips                    — upload audio clip (multipart)
GET  /api/clips                    — list all clips
GET  /api/clips/{id}/audio         — stream raw audio file
POST /api/runs                     — start a synchronous single-adapter run
GET  /api/runs/{id}                — fetch a completed run by id
WS   /ws/run/{run_id}              — stream StageStarted/StagePartial/
                                     StageCompleted/StageFailed events
"""
from __future__ import annotations

import asyncio
import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()  # load backend/.env if present

# ─── Adapter registry ─────────────────────────────────────────────────────────
# Each adapter module is imported defensively: if its heavy deps aren't
# installed (e.g. resemblyzer requires Python <3.12 because of numba),
# we log and skip rather than crashing the whole backend. The user can
# still exercise the adapters whose deps DID install.

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("trial-app")

from adapters.base import registry

_ADAPTERS = [
    ("adapters.deepgram_adapter", "DeepgramAdapter"),
    ("adapters.faster_whisper_adapter", "FasterWhisperAdapter"),
    ("adapters.gladia_adapter", "GladiaAdapter"),
    ("adapters.assemblyai_adapter", "AssemblyAIAdapter"),
    ("adapters.speechmatics_adapter", "SpeechmaticsAdapter"),
    ("adapters.groq_whisper_adapter", "GroqWhisperAdapter"),
    ("adapters.cartesia_tts_adapter", "CartesiaTTSAdapter"),
    ("adapters.resemblyzer_adapter", "ResemblyzerAdapter"),
    ("adapters.pyannote_verify_adapter", "PyannoteVerifyAdapter"),
    ("adapters.parakeet_adapter", "ParakeetAdapter"),
    ("adapters.canary_1b_flash_adapter", "Canary1BFlashAdapter"),
    ("adapters.canary_qwen_25b_adapter", "CanaryQwen25BAdapter"),
    # Slow-loop adapters (Slice 9.1c)
    ("adapters.qwen_intent_adapter", "QwenIntentAdapter"),
    ("adapters.haoclaw_outbox_adapter", "HaoClawOutboxAdapter"),
    # LID adapters for slow-loop-routed (Slice 9.2)
    ("adapters.whisper_lid_adapter", "WhisperLIDAdapter"),
    ("adapters.deepgram_lid_adapter", "DeepgramLIDAdapter"),
]

for mod_path, cls_name in _ADAPTERS:
    try:
        mod = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        registry.register(cls())
        log.info(f"registered adapter: {cls_name}")
    except Exception as e:
        log.warning(
            f"skipping adapter {cls_name}: {type(e).__name__}: {e}"
        )

# ─── Storage ──────────────────────────────────────────────────────────────────
from storage.clips import (
    Clip, FFmpegMissingError, audio_path, get_clip, list_clips,
    new_clip_id, save_clip, source_path,
)
from storage.runs import Run, append_run, get_run, new_run_id

# ─── Pipelines ────────────────────────────────────────────────────────────────
from pipelines.recipes import get_recipe, list_recipes
from pipelines.runner import run_pipeline

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="audio-trial-backend", version="0.1.0")

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")
PUBLIC_ORIGIN = os.environ.get("PUBLIC_ORIGIN", "")
allow_origins = [FRONTEND_ORIGIN, "http://localhost:3000", "http://localhost:5173"]
if PUBLIC_ORIGIN:
    allow_origins.append(PUBLIC_ORIGIN)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(allow_origins)),  # dedupe, keep order
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static frontend mount happens at the END of this file so it does NOT
# shadow /api/* and /ws/* routes — Starlette matches routes in order.

# In-memory map of run_id → WebSocket (only one socket per run in Slice 0)
_ws_connections: Dict[str, WebSocket] = {}

# Per-run event buffer so streaming events emitted before the client's
# WebSocket connects aren't lost. On WS accept we replay this buffer once,
# then deliver live events. Buffers self-clean when the WS closes OR when
# a `StageCompleted`/`StageFailed`/`RunFinished` event lands and stays
# unread for >30 s.
_event_buffer: Dict[str, list] = {}


# ─── WS event helpers ─────────────────────────────────────────────────────────

async def _ws_send(run_id: str, event: dict) -> None:
    """Emit an event to the run's WS subscriber; buffer it either way so a
    late-connecting client gets the full timeline on connect."""
    _event_buffer.setdefault(run_id, []).append(event)
    ws = _ws_connections.get(run_id)
    if ws:
        try:
            await ws.send_json(event)
        except Exception:
            pass


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/adapters")
async def list_adapters(category: str | None = None):
    """List registered adapters. Optional ?category=asr|tts|speaker_verify filter.

    Returns a flat list (REST convention) — frontend expects `Adapter[]` directly.
    """
    items = registry.to_json()
    if category:
        items = [a for a in items if a.get("category") == category]
    return items


# ── Clips ─────────────────────────────────────────────────────────────────────

class ClipOut(BaseModel):
    id: str
    source: str
    modality: str
    filename: str
    format: str
    original_filename: str = ""
    original_format: str = ""
    duration_s: float
    sample_rate: int
    channels: int
    language_detected: Optional[str]
    snr_db: Optional[float]
    speaker_count_estimate: Optional[int]
    user_tags: list
    scenarios: list
    uploaded_by: str
    created_at: str
    # Plan D A2 — ground-truth-seed transcript captured at /ws/mic save time.
    # null/empty for clips that came from upload or record-blob paths.
    captured_transcript: Optional[str] = None
    captured_transcript_segments: list = []


@app.post("/api/clips", response_model=ClipOut, status_code=201)
async def upload_clip(
    file: UploadFile = File(...),
    source: str = Form("upload"),
    uploaded_by: str = Form(""),
):
    """Accept an audio OR video file. Video containers (.mp4, .mov, .webm, .mkv,
    .avi, .m4v, .ts, .mts) get their audio track extracted to wav-16k-mono via
    ffmpeg; the original is kept adjacent for reference. Adapters always read
    the canonical wav.
    """
    upload_bytes = await file.read()
    original_filename = file.filename or "audio.wav"
    ext = Path(original_filename).suffix.lstrip(".").lower() or "wav"

    clip_id = new_clip_id()
    now = datetime.datetime.utcnow().isoformat() + "Z"

    clip = Clip(
        id=clip_id,
        source=source,
        modality="audio",
        original_filename=original_filename,
        original_format=ext,
        uploaded_by=uploaded_by,
        created_at=now,
    )
    try:
        clip = save_clip(clip, upload_bytes, ext)
    except FFmpegMissingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        # Bad/unreadable container, ffmpeg failure, etc.
        raise HTTPException(status_code=400,
                            detail=f"could not extract audio: {e}")

    # Read metadata from the CANONICAL audio (post-extraction for video).
    canonical = audio_path(clip_id)
    if canonical and canonical.exists():
        try:
            import soundfile as sf
            info = sf.info(str(canonical))
            clip.duration_s = float(info.duration)
            clip.sample_rate = int(info.samplerate)
            clip.channels = int(info.channels)
        except Exception:
            pass
        # Re-write manifest with the metadata filled in.
        from storage.clips import _clip_dir   # type: ignore
        import json as _json
        (_clip_dir(clip_id) / "manifest.json").write_text(
            _json.dumps(clip.to_dict(), indent=2), encoding="utf-8"
        )
    return ClipOut(**clip.to_dict())


@app.get("/api/clips", response_model=Dict[str, Any])
async def get_clips():
    clips = list_clips()
    return {"clips": [c.to_dict() for c in clips]}


class ClipPatch(BaseModel):
    user_tags: Optional[list] = None
    scenarios: Optional[list] = None
    # Auto-tagger fields — populated by the ingest script (or a future
    # auto-tagger in the upload route).  Not normally set from the UI.
    language_detected: Optional[str] = None
    snr_db: Optional[float] = None
    speaker_count_estimate: Optional[int] = None
    # Plan D A2 — let users hand-correct the captured transcript so the
    # AR-glass benchmark can use it as a ground-truth reference.
    captured_transcript: Optional[str] = None
    captured_transcript_segments: Optional[list] = None


@app.patch("/api/clips/{clip_id}", response_model=ClipOut)
async def update_clip(clip_id: str, patch: ClipPatch):
    """Mutate a clip's user_tags / scenarios + auto-tagger metadata —
    used by the Corpus page chip tagger and the ingest auto-tagger.
    Other fields (id / format / duration_s / etc.) are immutable
    post-ingest by design."""
    import json as _json
    from storage.clips import _clip_dir   # type: ignore
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if patch.user_tags is not None:
        clip.user_tags = list(dict.fromkeys(patch.user_tags))   # dedupe
    if patch.scenarios is not None:
        clip.scenarios = list(dict.fromkeys(patch.scenarios))
    if patch.language_detected is not None:
        clip.language_detected = patch.language_detected
    if patch.snr_db is not None:
        clip.snr_db = float(patch.snr_db)
    if patch.speaker_count_estimate is not None:
        clip.speaker_count_estimate = int(patch.speaker_count_estimate)
    if patch.captured_transcript is not None:
        clip.captured_transcript = patch.captured_transcript
    if patch.captured_transcript_segments is not None:
        clip.captured_transcript_segments = list(patch.captured_transcript_segments)
    (_clip_dir(clip_id) / "manifest.json").write_text(
        _json.dumps(clip.to_dict(), indent=2), encoding="utf-8"
    )
    return ClipOut(**clip.to_dict())


@app.delete("/api/clips/{clip_id}", status_code=204)
async def delete_clip(clip_id: str):
    from storage.clips import _clip_dir   # type: ignore
    import shutil
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    shutil.rmtree(_clip_dir(clip_id), ignore_errors=True)
    return None


# ── Auto-tag scenarios ───────────────────────────────────────────────────────
# Heuristic acoustic tagger that looks at SNR, spectral centroid, sample-rate
# bandwidth, plus optional Whisper-LID + Resemblyzer signals to assign a
# subset of the SCENARIO_PALETTE to each clip without the user having to
# click chips one-by-one in the Corpus page. See pipelines/auto_tagger.py
# for the rule set.

class AutoTagOptions(BaseModel):
    use_lid: bool = True
    use_speaker_spread: bool = True
    # If True, replace existing scenarios entirely; if False (default), union
    # the auto-detected tags with whatever the user already set so manual
    # work isn't clobbered.
    replace: bool = False


class AutoTagResult(BaseModel):
    clip_id: str
    detected: list                   # scenarios produced by the tagger
    final_scenarios: list            # what the clip ended up with after merge
    features: Dict[str, Any]
    evidence: Dict[str, str]


@app.post("/api/clips/{clip_id}/autotag", response_model=AutoTagResult)
async def autotag_clip_endpoint(clip_id: str, opts: AutoTagOptions | None = None):
    """Run the heuristic auto-tagger on a single clip and persist the
    detected scenarios to its manifest. Returns the detected list +
    feature values + per-tag evidence so the UI can show "why"."""
    from pipelines.auto_tagger import autotag_clip
    from storage.clips import _clip_dir   # type: ignore
    import json as _json

    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    p = audio_path(clip_id)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    options = opts or AutoTagOptions()
    try:
        result = await autotag_clip(
            str(p),
            registry=registry,
            use_lid=options.use_lid,
            use_speaker_spread=options.use_speaker_spread,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"autotag failed: {e}")

    detected = result.get("scenarios") or []
    if options.replace:
        clip.scenarios = list(dict.fromkeys(detected))
    else:
        clip.scenarios = list(dict.fromkeys([*clip.scenarios, *detected]))

    (_clip_dir(clip_id) / "manifest.json").write_text(
        _json.dumps(clip.to_dict(), indent=2), encoding="utf-8"
    )
    return AutoTagResult(
        clip_id=clip_id,
        detected=detected,
        final_scenarios=clip.scenarios,
        features=result.get("features") or {},
        evidence=result.get("evidence") or {},
    )


class AutoTagAllResult(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list                    # list of AutoTagResult-shaped dicts
    errors: Dict[str, str]           # clip_id → error message (only failures)


@app.post("/api/clips/autotag-all", response_model=AutoTagAllResult)
async def autotag_all_clips(opts: AutoTagOptions | None = None):
    """Run the heuristic auto-tagger on EVERY clip in the corpus. The
    Corpus page's 'Auto-tag all' button hits this; clips run sequentially
    so we don't thrash CPU when the user has 100+ clips. Per-clip failures
    don't abort the batch — they're returned in the `errors` map so the
    UI can flag them."""
    from pipelines.auto_tagger import autotag_clip
    from storage.clips import _clip_dir, list_clips   # type: ignore
    import json as _json

    options = opts or AutoTagOptions()
    clips = list_clips()
    results: list = []
    errors: Dict[str, str] = {}

    for clip in clips:
        p = audio_path(clip.id)
        if not p or not p.exists():
            errors[clip.id] = "audio file missing"
            continue
        try:
            result = await autotag_clip(
                str(p),
                registry=registry,
                use_lid=options.use_lid,
                use_speaker_spread=options.use_speaker_spread,
            )
        except Exception as e:
            errors[clip.id] = f"{type(e).__name__}: {e}"
            continue

        detected = result.get("scenarios") or []
        if options.replace:
            clip.scenarios = list(dict.fromkeys(detected))
        else:
            clip.scenarios = list(dict.fromkeys([*clip.scenarios, *detected]))

        (_clip_dir(clip.id) / "manifest.json").write_text(
            _json.dumps(clip.to_dict(), indent=2), encoding="utf-8"
        )
        results.append({
            "clip_id": clip.id,
            "detected": detected,
            "final_scenarios": clip.scenarios,
            "features": result.get("features") or {},
            "evidence": result.get("evidence") or {},
        })

    return AutoTagAllResult(
        total=len(clips),
        succeeded=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )


@app.get("/api/clips/{clip_id}/audio")
async def stream_clip_audio(clip_id: str):
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    p = audio_path(clip_id)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    media_types = {
        "wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg",
        "ogg": "audio/ogg", "flac": "audio/flac", "m4a": "audio/mp4",
    }
    media_type = media_types.get(clip.format, "application/octet-stream")
    return FileResponse(str(p), media_type=media_type)


# ── Enrollment (Slice 9.1e) ───────────────────────────────────────────────────
# The wearer's reference embedding lives at data/enrollments/wearer.json.
# Slow-loop's speaker_tag stage reads this on every run via stage config.
# v2 (familiar voices) and v3 (auto-enroll cluster IDs) extend the same
# directory with one file per profile; no schema migration needed.

class EnrollOut(BaseModel):
    profile_id: str
    adapter: str
    embedding_dim: int
    embedding_dtype: str
    duration_s: float | None = None
    saved_to: str


class EnrollListItem(BaseModel):
    profile_id: str
    adapter: str
    embedding_dim: int
    saved_to: str
    enrolled_at: str


def _enrollments_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data")) / "enrollments"


@app.post("/api/enroll", response_model=EnrollOut, status_code=201)
async def enroll_wearer(
    file: UploadFile = File(...),
    adapter: str = Form("pyannote_verify"),
    profile_id: str = Form("wearer"),
):
    """Enroll a reference clip → embedding → JSON on disk.

    Audio is saved to a tempfile, the chosen speaker_verify adapter's
    `enroll()` method is called, and the resulting embedding (plus metadata)
    is persisted to `data/enrollments/<profile_id>.json` so the slow-loop
    speaker_tag stage can load it without re-uploading.
    """
    try:
        ad = registry.get(adapter)
    except KeyError:
        raise HTTPException(status_code=400,
                            detail=f"adapter {adapter!r} not registered")
    if getattr(ad, "category", None) != "speaker_verify":
        raise HTTPException(
            status_code=400,
            detail=f"adapter {adapter!r} is category "
                   f"{getattr(ad, 'category', '?')!r}, expected speaker_verify",
        )

    audio_bytes = await file.read()
    suffix = Path(file.filename or "ref.wav").suffix or ".wav"
    import tempfile as _tempfile
    with _tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = await ad.enroll(tmp_path, {})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    enr_dir = _enrollments_dir()
    enr_dir.mkdir(parents=True, exist_ok=True)
    saved_to = enr_dir / f"{profile_id}.json"

    # Best-effort duration from the canonical audio
    duration_s: Optional[float] = None
    try:
        import io as _io
        import soundfile as _sf
        info = _sf.info(_io.BytesIO(audio_bytes))
        duration_s = float(info.duration)
    except Exception:
        pass

    record = {
        "profile_id": profile_id,
        "adapter": adapter,
        "embedding_b64": result["embedding_b64"],
        "embedding_dim": result["embedding_dim"],
        "embedding_dtype": result.get("embedding_dtype", "float32"),
        "duration_s": duration_s,
        "enrolled_at": datetime.datetime.utcnow().isoformat() + "Z",
        "filename": file.filename,
    }
    saved_to.write_text(__import__("json").dumps(record, indent=2),
                        encoding="utf-8")

    return EnrollOut(
        profile_id=profile_id,
        adapter=adapter,
        embedding_dim=record["embedding_dim"],
        embedding_dtype=record["embedding_dtype"],
        duration_s=duration_s,
        saved_to=str(saved_to),
    )


@app.get("/api/enroll", response_model=Dict[str, Any])
async def list_enrollments():
    """List enrolled profiles. Reads data/enrollments/*.json metadata."""
    import json as _json
    out = []
    d = _enrollments_dir()
    if d.exists():
        for f in sorted(d.glob("*.json")):
            try:
                rec = _json.loads(f.read_text(encoding="utf-8"))
                out.append({
                    "profile_id": rec["profile_id"],
                    "adapter": rec["adapter"],
                    "embedding_dim": rec["embedding_dim"],
                    "saved_to": str(f),
                    "enrolled_at": rec.get("enrolled_at", ""),
                })
            except Exception:
                pass
    return {"enrollments": out}


@app.delete("/api/enroll/{profile_id}", status_code=204)
async def delete_enrollment(profile_id: str):
    """Delete a saved enrollment so the wearer can re-enroll."""
    p = _enrollments_dir() / f"{profile_id}.json"
    if p.exists():
        try:
            p.unlink()
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e))
    return None


@app.get("/api/settings")
async def get_settings():
    """Read-only env-var status for the Settings page.

    Reports which API keys + service URLs are configured WITHOUT exposing
    secret values. Used to render green/grey 'configured' chips beside
    each adapter category.
    """
    keys = [
        "DEEPGRAM_API_KEY", "GLADIA_API_KEY", "ASSEMBLYAI_API_KEY",
        "SPEECHMATICS_API_KEY", "GROQ_API_KEY", "CARTESIA_API_KEY",
        "ELEVENLABS_API_KEY", "OPENAI_API_KEY", "HF_TOKEN",
    ]
    urls = ["INTENT_LLM_URL", "MODEL_SERVER_URL", "PUBLIC_ORIGIN", "DATA_DIR"]
    api_keys: Dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k, "")
        if v and not v.startswith("your_"):
            api_keys[k] = "set"
        else:
            api_keys[k] = "unset"

    url_status: Dict[str, Any] = {}
    for k in urls:
        v = os.environ.get(k, "")
        url_status[k] = {"value": v if v else None,
                         "configured": bool(v) and not v.startswith("your_")}

    return {
        "api_keys": api_keys,
        "service_urls": url_status,
        "intent_llm": {
            "url_configured": bool(os.environ.get("INTENT_LLM_URL")),
            "default_model": os.environ.get(
                "INTENT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            "key_configured": bool(os.environ.get("INTENT_LLM_KEY")),
        },
    }


@app.get("/api/enroll/{profile_id}/embedding")
async def get_enrollment_embedding(profile_id: str):
    """Return the raw embedding_b64 for a profile — used by the slow-loop
    speaker_tag stage to feed verify_segments() without round-tripping
    through the client."""
    import json as _json
    p = _enrollments_dir() / f"{profile_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404,
                            detail=f"profile {profile_id!r} not enrolled")
    rec = _json.loads(p.read_text(encoding="utf-8"))
    return {
        "profile_id": rec["profile_id"],
        "adapter": rec["adapter"],
        "embedding_b64": rec["embedding_b64"],
        "embedding_dim": rec["embedding_dim"],
    }


# ── Runs ──────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    clip_id: str
    adapter: str
    config: Dict[str, Any] = {}


class RunOut(BaseModel):
    id: str
    clip_id: str
    adapter: str
    config: Dict[str, Any]
    started_at: str
    finished_at: Optional[str]
    latency_ms: Optional[float]
    cost_usd: Optional[float]
    input_preview: str
    output_preview: str
    result: Dict[str, Any]
    error: Optional[str]


# In-memory cache of in-flight runs so GET /api/runs/{id} works mid-stream.
_active_runs: Dict[str, Run] = {}


async def _execute_run(run: Run, adapter: Any, audio_file: Path) -> None:
    """Run an adapter against an audio file, streaming progress events to
    the per-run WS. Updates the run object in place; persists via append_run
    only on completion or failure."""
    run_id = run.id
    is_streaming = bool(getattr(adapter, "is_streaming", False))
    t0 = time.perf_counter()
    req_config = run.config

    try:
        if is_streaming:
            result: Dict[str, Any] = {}
            partial_count = 0
            async for ev in adapter.transcribe_stream(str(audio_file), req_config):
                if ev.get("is_final"):
                    result = {k: v for k, v in ev.items()
                              if k not in ("is_final", "partial_text")}
                    result.setdefault("text", ev.get("partial_text", ""))
                    break
                partial_count += 1
                await _ws_send(run_id, {
                    "event": "StageProgress",
                    "run_id": run_id,
                    "adapter": run.adapter,
                    "partial_text": ev.get("partial_text", ""),
                    "partial_index": partial_count,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                })
            if not result:
                raise RuntimeError("streaming adapter ended without is_final yield")
        else:
            result = await adapter.transcribe(str(audio_file), req_config)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        finished_at = datetime.datetime.utcnow().isoformat() + "Z"

        run.finished_at = finished_at
        run.latency_ms = latency_ms
        run.cost_usd = result.get("cost_usd")
        run.input_preview = audio_file.name
        run.output_preview = (result.get("text", "") or "")[:200]
        run.result = {k: v for k, v in result.items() if k != "raw_response"}
        run.raw_response = result.get("raw_response")

        if not is_streaming:
            await _ws_send(run_id, {
                "event": "StagePartial",
                "run_id": run_id,
                "adapter": run.adapter,
                "text": run.output_preview,
                "timestamp": finished_at,
            })

        await _ws_send(run_id, {
            "event": "StageCompleted",
            "run_id": run_id,
            "adapter": run.adapter,
            "latency_ms": latency_ms,
            "cost_usd": run.cost_usd,
            "result": run.result,
            "timestamp": finished_at,
            "is_streaming": is_streaming,
        })

    except Exception as exc:
        finished_at = datetime.datetime.utcnow().isoformat() + "Z"
        run.finished_at = finished_at
        run.error = f"{type(exc).__name__}: {exc}"
        await _ws_send(run_id, {
            "event": "StageFailed",
            "run_id": run_id,
            "adapter": run.adapter,
            "error": run.error,
            "timestamp": finished_at,
        })

    append_run(run)
    _active_runs.pop(run_id, None)


@app.post("/api/runs", response_model=RunOut, status_code=201)
async def create_run(req: RunRequest):
    # Validate clip
    clip = get_clip(req.clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail=f"Clip {req.clip_id!r} not found")

    # Validate adapter
    try:
        adapter = registry.get(req.adapter)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Adapter {req.adapter!r} not registered")

    p = audio_path(req.clip_id)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Audio file for clip not found")

    run_id = new_run_id()
    now = datetime.datetime.utcnow().isoformat() + "Z"
    run = Run(
        id=run_id,
        clip_id=req.clip_id,
        adapter=req.adapter,
        config=req.config,
        started_at=now,
    )
    _active_runs[run_id] = run

    await _ws_send(run_id, {
        "event": "StageStarted",
        "run_id": run_id,
        "adapter": req.adapter,
        "timestamp": now,
    })

    is_streaming = bool(getattr(adapter, "is_streaming", False))

    if is_streaming:
        # Streaming: kick off the work in the background and return the run
        # record immediately so the client can open WS to watch live partials.
        # The response carries the placeholder run; GET /api/runs/{id} or the
        # WS StageCompleted event delivers the final result.
        asyncio.create_task(_execute_run(run, adapter, p))
        return RunOut(**run.to_dict())

    # Batch path — execute synchronously so the response carries the result.
    await _execute_run(run, adapter, p)
    return RunOut(**run.to_dict())


@app.get("/api/runs/{run_id}", response_model=RunOut)
async def fetch_run(run_id: str):
    # Try the active in-memory record first (mid-stream runs); fall back to
    # the persisted JSONL store after the run finishes.
    if run_id in _active_runs:
        return RunOut(**_active_runs[run_id].to_dict())
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunOut(**run.to_dict())


# ── Recipes ───────────────────────────────────────────────────────────────────

@app.get("/api/recipes")
async def get_recipes():
    """List built-in pipeline recipes (Slice 5).

    Each recipe declares stages with category placeholders; the client picks
    a concrete adapter per stage when starting a run.
    """
    return list_recipes()


@app.get("/api/recipes/{recipe_id}")
async def get_recipe_one(recipe_id: str):
    r = get_recipe(recipe_id)
    if not r:
        raise HTTPException(status_code=404, detail=f"Recipe {recipe_id!r} not found")
    return r


class RecipeRunRequest(BaseModel):
    clip_id: str
    recipe_id: str
    # stage_id → adapter_id (resolves the recipe's `adapter: None` placeholders)
    stage_adapters: Dict[str, str]
    # Optional per-stage config override
    stage_configs: Dict[str, Dict[str, Any]] = {}


class RecipeRunOut(BaseModel):
    id: str
    clip_id: str
    recipe_id: str
    started_at: str
    finished_at: str
    stages: list
    total_latency_ms: float
    total_cost_usd: float
    error: Optional[str]


@app.post("/api/runs/recipe", response_model=RecipeRunOut, status_code=201)
async def create_recipe_run(req: RecipeRunRequest):
    """Run a multi-stage recipe pipeline against a clip.

    Per-stage adapter override is mandatory because recipes ship with
    `adapter: None` placeholders. The runner walks stages in order, threading
    each stage's output into the next (ASR text → TTS input, etc).
    Per-stage events stream over /ws/run/{id}.
    """
    clip = get_clip(req.clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail=f"Clip {req.clip_id!r} not found")

    p = audio_path(req.clip_id)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Audio file for clip not found")

    recipe = get_recipe(req.recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail=f"Recipe {req.recipe_id!r} not found")

    run_id = new_run_id()
    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    async def _on_event(payload: Dict[str, Any]) -> None:
        await _ws_send(run_id, {**payload, "run_id": run_id})

    await _on_event({"event": "RunStarted", "recipe_id": req.recipe_id})

    out = await run_pipeline(
        pipeline=recipe,
        clip_audio_path=str(p),
        stage_overrides=req.stage_adapters,
        stage_configs=req.stage_configs,
        registry=registry,
        on_event=_on_event,
    )

    finished_at = datetime.datetime.utcnow().isoformat() + "Z"
    await _on_event({"event": "RunFinished",
                     "total_latency_ms": out["total_latency_ms"],
                     "total_cost_usd": out["total_cost_usd"],
                     "error": out["error"]})

    return RecipeRunOut(
        id=run_id,
        clip_id=req.clip_id,
        recipe_id=req.recipe_id,
        started_at=started_at,
        finished_at=finished_at,
        stages=out["stages"],
        total_latency_ms=out["total_latency_ms"],
        total_cost_usd=out["total_cost_usd"],
        error=out["error"],
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/run/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str):
    """Stream per-stage events for a run.

    Race handling: a client may connect AFTER the run has started (or even
    after it's finished, for short batch runs). We replay the per-run
    event buffer first thing on connect, then forward live events for the
    rest of the session.
    """
    await websocket.accept()
    _ws_connections[run_id] = websocket
    try:
        # Replay anything already emitted before the WS opened
        for ev in list(_event_buffer.get(run_id, [])):
            try:
                await websocket.send_json(ev)
            except Exception:
                break

        # Then pump keepalives / forward live events
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                if msg == "ping":
                    await websocket.send_json({"event": "pong"})
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"event": "keepalive"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        _ws_connections.pop(run_id, None)


# ─── Live-mic streaming (Slice 4) ──────────────────────────────────────────────
# Bidirectional WS proxy: the browser sends Int16 PCM @ 16 kHz, we forward it
# to the vendor's streaming WS, and translate the vendor's partial-transcript
# events back into our common StageProgress / StageCompleted shape.
#
# Client protocol (frontend):
#   binary frames  = raw PCM s16le (any chunk size)
#   text frames    = JSON control:
#                      {"type": "stop"}
#
# Server emits to client:
#   {event: "StageStarted", adapter, ...}
#   {event: "StageProgress", partial_text, partial_index}
#   {event: "StageCompleted", result: {text, words, ...}, latency_ms}
#   {event: "StageFailed", error}

@app.websocket("/ws/mic")
async def ws_mic(websocket: WebSocket, adapter: str, sample_rate: int = 16000):
    await websocket.accept()
    try:
        adapter_obj = registry.get(adapter)
    except KeyError:
        await websocket.send_json({"event": "StageFailed",
                                   "error": f"Adapter {adapter!r} not registered"})
        await websocket.close()
        return

    if not getattr(adapter_obj, "is_streaming", False):
        await websocket.send_json({
            "event": "StageFailed",
            "error": f"Adapter {adapter!r} is not streaming-capable; "
                     f"use POST /api/runs for batch adapters",
        })
        await websocket.close()
        return

    if adapter in ("deepgram",):
        await _proxy_deepgram_mic(websocket, sample_rate, adapter_obj)
    elif adapter in ("assemblyai",):
        await _proxy_assemblyai_mic(websocket, sample_rate, adapter_obj)
    else:
        await websocket.send_json({
            "event": "StageFailed",
            "error": f"Adapter {adapter!r} has no mic-streaming proxy yet",
        })
        await websocket.close()


async def _proxy_deepgram_mic(client_ws: WebSocket, sample_rate: int, adapter_obj: Any) -> None:
    """Forward PCM from client → Deepgram WS; translate Results frames back."""
    import json as _json
    import websockets as wslib

    api_key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not api_key:
        await client_ws.send_json({"event": "StageFailed",
                                   "error": "DEEPGRAM_API_KEY not set"})
        await client_ws.close(); return

    params = {
        "model": "nova-3",
        "language": "en",
        "encoding": "linear16",
        "sample_rate": str(sample_rate),
        "channels": "1",
        "smart_format": "true",
        "punctuate": "true",
        "interim_results": "true",
    }
    url = "wss://api.deepgram.com/v1/listen?" + "&".join(
        f"{k}={v}" for k, v in params.items()
    )

    t0 = time.perf_counter()
    final_segments: list = []
    latest_partial = ""
    full_words: list = []
    partial_count = 0
    stopped = False

    try:
        async with wslib.connect(
            url,
            additional_headers={"Authorization": f"Token {api_key}"},
            max_size=None,
        ) as vendor_ws:
            await client_ws.send_json({
                "event": "StageStarted",
                "adapter": "deepgram",
                "sample_rate": sample_rate,
            })

            async def client_to_vendor() -> None:
                """Pump bytes from browser → Deepgram, until client says stop."""
                nonlocal stopped
                while True:
                    msg = await client_ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        stopped = True
                        return
                    if "bytes" in msg and msg["bytes"]:
                        await vendor_ws.send(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        try:
                            ctrl = _json.loads(msg["text"])
                        except Exception:
                            continue
                        if ctrl.get("type") == "stop":
                            stopped = True
                            await vendor_ws.send(_json.dumps({"type": "CloseStream"}))
                            return

            async def vendor_to_client() -> None:
                nonlocal partial_count, latest_partial
                async for raw in vendor_ws:
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
                            })
                    else:
                        latest_partial = text
                    accumulated = " ".join(final_segments)
                    if latest_partial:
                        accumulated = (accumulated + " " + latest_partial).strip()
                    partial_count += 1
                    await client_ws.send_json({
                        "event": "StageProgress",
                        "partial_text": accumulated,
                        "partial_index": partial_count,
                    })

            send_task = asyncio.create_task(client_to_vendor())
            recv_task = asyncio.create_task(vendor_to_client())
            done, pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            for t in pending:
                try: await t
                except Exception: pass

        full_text = " ".join(final_segments).strip()
        wall_s = time.perf_counter() - t0
        await client_ws.send_json({
            "event": "StageCompleted",
            "adapter": "deepgram",
            "latency_ms": wall_s * 1000.0,
            "result": {
                "text": full_text,
                "words": full_words,
                "language": "en",
                "wall_time_s": wall_s,
            },
            "is_streaming": True,
        })
    except Exception as exc:
        await client_ws.send_json({
            "event": "StageFailed",
            "adapter": "deepgram",
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        try: await client_ws.close()
        except Exception: pass


async def _proxy_assemblyai_mic(client_ws: WebSocket, sample_rate: int, adapter_obj: Any) -> None:
    """Forward PCM from client → AssemblyAI v3 streaming WS; translate Turn frames."""
    import json as _json
    import websockets as wslib

    api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not api_key:
        await client_ws.send_json({"event": "StageFailed",
                                   "error": "ASSEMBLYAI_API_KEY not set"})
        await client_ws.close(); return

    url = (f"wss://streaming.assemblyai.com/v3/ws"
           f"?sample_rate={sample_rate}"
           f"&speech_model=universal-streaming-english"
           f"&format_turns=true")

    t0 = time.perf_counter()
    finalized_turns: list = []
    unfmt_partial = ""
    all_words: list = []
    partial_count = 0

    try:
        async with wslib.connect(
            url,
            additional_headers={"Authorization": api_key},
            max_size=None,
        ) as vendor_ws:
            await client_ws.send_json({
                "event": "StageStarted",
                "adapter": "assemblyai",
                "sample_rate": sample_rate,
            })

            async def client_to_vendor() -> None:
                while True:
                    msg = await client_ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if "bytes" in msg and msg["bytes"]:
                        await vendor_ws.send(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        try:
                            ctrl = _json.loads(msg["text"])
                        except Exception:
                            continue
                        if ctrl.get("type") == "stop":
                            await vendor_ws.send(_json.dumps({"type": "Terminate"}))
                            return

            async def vendor_to_client() -> None:
                nonlocal partial_count, unfmt_partial
                async for raw in vendor_ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        msg = _json.loads(raw)
                    except _json.JSONDecodeError:
                        continue
                    mtype = msg.get("type")
                    if mtype in ("Begin",):
                        continue
                    if mtype == "Termination":
                        return
                    if mtype != "Turn":
                        continue
                    transcript = msg.get("transcript", "") or ""
                    end_of_turn = bool(msg.get("end_of_turn"))
                    is_formatted = bool(msg.get("turn_is_formatted"))
                    if end_of_turn and is_formatted:
                        if transcript:
                            finalized_turns.append(transcript)
                        unfmt_partial = ""
                        for w in msg.get("words", []) or []:
                            all_words.append({
                                "word": w.get("text", ""),
                                "start": float(w.get("start", 0)) / 1000.0,
                                "end": float(w.get("end", 0)) / 1000.0,
                            })
                    else:
                        unfmt_partial = transcript
                    accumulated = " ".join(finalized_turns)
                    if unfmt_partial:
                        accumulated = (accumulated + " " + unfmt_partial).strip()
                    partial_count += 1
                    await client_ws.send_json({
                        "event": "StageProgress",
                        "partial_text": accumulated,
                        "partial_index": partial_count,
                    })

            send_task = asyncio.create_task(client_to_vendor())
            recv_task = asyncio.create_task(vendor_to_client())
            done, pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            for t in pending:
                try: await t
                except Exception: pass

        full_text = " ".join(finalized_turns).strip()
        wall_s = time.perf_counter() - t0
        await client_ws.send_json({
            "event": "StageCompleted",
            "adapter": "assemblyai",
            "latency_ms": wall_s * 1000.0,
            "result": {
                "text": full_text,
                "words": all_words,
                "language": "en",
                "wall_time_s": wall_s,
            },
            "is_streaming": True,
        })
    except Exception as exc:
        await client_ws.send_json({
            "event": "StageFailed",
            "adapter": "assemblyai",
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        try: await client_ws.close()
        except Exception: pass


# ─── Static frontend mount (production, MUST be last) ────────────────────────
# In docker-compose deploys the multi-stage Dockerfile bakes
# frontend/dist into /app/frontend_dist. When that directory exists we
# mount it on / so the SPA and the API share an origin (Caddy forwards
# everything to one backend). In dev (pnpm dev on :5173) the directory
# is absent and this is a no-op.
_FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", "/app/frontend_dist"))
if _FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True),
              name="frontend")
