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
from adapters.base import registry
from adapters.deepgram_adapter import DeepgramAdapter
from adapters.faster_whisper_adapter import FasterWhisperAdapter
from adapters.gladia_adapter import GladiaAdapter
from adapters.assemblyai_adapter import AssemblyAIAdapter
from adapters.speechmatics_adapter import SpeechmaticsAdapter
from adapters.groq_whisper_adapter import GroqWhisperAdapter
from adapters.cartesia_tts_adapter import CartesiaTTSAdapter
from adapters.resemblyzer_adapter import ResemblyzerAdapter
from adapters.pyannote_verify_adapter import PyannoteVerifyAdapter
from adapters.parakeet_adapter import ParakeetAdapter
from adapters.canary_1b_flash_adapter import Canary1BFlashAdapter
from adapters.canary_qwen_25b_adapter import CanaryQwen25BAdapter

registry.register(DeepgramAdapter())
registry.register(FasterWhisperAdapter())
registry.register(GladiaAdapter())
registry.register(AssemblyAIAdapter())
registry.register(SpeechmaticsAdapter())
registry.register(GroqWhisperAdapter())
registry.register(CartesiaTTSAdapter())
registry.register(ResemblyzerAdapter())
registry.register(PyannoteVerifyAdapter())
registry.register(ParakeetAdapter())
registry.register(Canary1BFlashAdapter())
registry.register(CanaryQwen25BAdapter())

# ─── Storage ──────────────────────────────────────────────────────────────────
from storage.clips import Clip, audio_path, get_clip, list_clips, new_clip_id, save_clip
from storage.runs import Run, append_run, get_run, new_run_id

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
    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"
    ext = Path(filename).suffix.lstrip(".").lower() or "wav"

    clip_id = new_clip_id()
    now = datetime.datetime.utcnow().isoformat() + "Z"

    # Best-effort metadata extraction (soundfile for WAV/FLAC/OGG; skip on error)
    duration_s = 0.0
    sample_rate = 0
    channels = 1
    try:
        import io
        import soundfile as sf
        info = sf.info(io.BytesIO(audio_bytes))
        duration_s = float(info.duration)
        sample_rate = int(info.samplerate)
        channels = int(info.channels)
    except Exception:
        pass

    clip = Clip(
        id=clip_id,
        source=source,
        modality="audio",
        filename=filename,
        format=ext,
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        uploaded_by=uploaded_by,
        created_at=now,
    )
    clip = save_clip(clip, audio_bytes, ext)
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
