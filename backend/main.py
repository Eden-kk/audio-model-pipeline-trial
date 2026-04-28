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


# ─── WS event helpers ─────────────────────────────────────────────────────────

async def _ws_send(run_id: str, event: dict) -> None:
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

    # Emit StageStarted
    await _ws_send(run_id, {
        "event": "StageStarted",
        "run_id": run_id,
        "adapter": req.adapter,
        "timestamp": now,
    })

    t0 = time.perf_counter()
    try:
        result = await adapter.transcribe(str(p), req.config)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        finished_at = datetime.datetime.utcnow().isoformat() + "Z"

        run.finished_at = finished_at
        run.latency_ms = latency_ms
        run.cost_usd = result.get("cost_usd")
        run.input_preview = str(p.name)
        run.output_preview = (result.get("text", "") or "")[:200]
        run.result = {k: v for k, v in result.items() if k != "raw_response"}
        run.raw_response = result.get("raw_response")

        # Emit StagePartial (transcript preview) then StageCompleted
        await _ws_send(run_id, {
            "event": "StagePartial",
            "run_id": run_id,
            "adapter": req.adapter,
            "text": run.output_preview,
            "timestamp": finished_at,
        })
        await _ws_send(run_id, {
            "event": "StageCompleted",
            "run_id": run_id,
            "adapter": req.adapter,
            "latency_ms": latency_ms,
            "cost_usd": run.cost_usd,
            "result": run.result,
            "timestamp": finished_at,
        })

    except Exception as exc:
        finished_at = datetime.datetime.utcnow().isoformat() + "Z"
        run.finished_at = finished_at
        run.error = f"{type(exc).__name__}: {exc}"
        await _ws_send(run_id, {
            "event": "StageFailed",
            "run_id": run_id,
            "adapter": req.adapter,
            "error": run.error,
            "timestamp": finished_at,
        })

    append_run(run)
    return RunOut(**run.to_dict())


@app.get("/api/runs/{run_id}", response_model=RunOut)
async def fetch_run(run_id: str):
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
    """Clients connect here *before* (or shortly after) POST /api/runs.
    Events are sent as JSON objects with an `event` field.
    """
    await websocket.accept()
    _ws_connections[run_id] = websocket
    try:
        # Keep the socket alive until the client closes it
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                if msg == "ping":
                    await websocket.send_json({"event": "pong"})
            except asyncio.TimeoutError:
                # send a keepalive ping
                try:
                    await websocket.send_json({"event": "keepalive"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        _ws_connections.pop(run_id, None)


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
