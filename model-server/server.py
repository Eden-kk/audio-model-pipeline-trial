"""model-server — FastAPI app hosting NeMo ASR models on AMD-ROCm.

Routes:
  GET  /health                       — liveness probe
  GET  /models                       — load status of every registered model
  POST /v1/transcribe?model=<id>     — multipart audio → JSON
                                       {text, words, language, duration_s,
                                        model, latency_ms}

The trial-app's NeMo adapters (parakeet, canary_1b_flash, canary_qwen_25b)
all hit POST /v1/transcribe and dispatch by ?model=.

Slice 1B status:
  - Parakeet: working end-to-end (loader + transcriber).
  - Canary-1B-flash: loader stubbed, returns 503 with a clear error.
  - Canary-Qwen-2.5B: same.
Slice 2 will fill in the Canary loaders once we can validate them on a
real AMD GPU.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

import model_loader

app = FastAPI(title="model-server", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0", "rocm": _rocm_present()}


def _rocm_present() -> bool:
    """Best-effort: report whether ROCm is visible in this container."""
    try:
        import torch  # type: ignore
        if hasattr(torch.version, "hip") and torch.version.hip:
            return True
        return torch.cuda.is_available()  # ROCm exposes this as True too
    except Exception:
        return False


@app.get("/models")
async def models():
    return {"models": model_loader.status(), "rocm": _rocm_present()}


@app.post("/v1/transcribe")
async def transcribe(
    model: str = Query(..., description="One of parakeet-tdt-1.1b, canary-1b-flash, canary-qwen-2.5b"),
    file: UploadFile = File(...),
):
    if model not in model_loader.LOADERS:
        raise HTTPException(status_code=400,
                            detail=f"unknown model id: {model!r}")

    audio_bytes = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        try:
            result = model_loader.transcribe(model, tmp_path)
        except NotImplementedError as e:
            # Loader/transcriber pending — surface as 503 with a useful body.
            raise HTTPException(status_code=503,
                                detail=f"model {model!r} not yet implemented: {e}")

        # Best-effort duration extraction
        try:
            import soundfile as sf  # type: ignore
            info = sf.info(tmp_path)
            result["duration_s"] = float(info.duration)
        except Exception:
            result.setdefault("duration_s", 0.0)

        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
