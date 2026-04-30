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

from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

import model_loader

app = FastAPI(title="model-server", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0", "rocm": _rocm_present()}


@app.get("/salm-debug")
async def salm_debug():
    """Dump everything we know about Canary-Qwen's prompt + audio config."""
    import inspect
    out: dict = {}
    try:
        m = model_loader.get_or_load("canary-qwen-2.5b")
        cfg = m.cfg
        out["prompt_format"] = getattr(cfg, "prompt_format", "?")
        # Look for any audio-locator-related keys
        for k in dir(cfg):
            if "audio" in k.lower() or "locator" in k.lower() or "placeholder" in k.lower():
                try:
                    out[f"cfg.{k}"] = str(getattr(cfg, k))[:200]
                except Exception:
                    pass
        # Probe the formatter for audio-placeholder slot/marker
        try:
            from nemo.collections.common.prompts.formatter import PromptFormatter
            fmt = PromptFormatter.resolve(cfg.prompt_format)(m.tokenizer)
            for attr in ["TEMPLATE", "_slots", "_format", "OUTPUT_ROLE", "INFERENCE_TURN"]:
                if hasattr(fmt, attr):
                    out[f"fmt.{attr}"] = str(getattr(fmt, attr))[:600]
            out["fmt.dir"] = [a for a in dir(fmt) if not a.startswith("_")][:30]
        except Exception as e:
            out["template_err"] = str(e)[:200]
        # Inspect SALM internal placeholder constants
        try:
            from nemo.collections.speechlm2.models import salm as _salm
            out["salm.audio_token"] = getattr(_salm, "AUDIO_LOCATOR_TAG", "?")
            for k in ["AUDIO_TOKEN", "PLACEHOLDER", "AUDIO_LOCATOR"]:
                if hasattr(_salm, k):
                    out[f"salm.{k}"] = str(getattr(_salm, k))
        except Exception as e:
            out["salm_consts_err"] = str(e)[:200]
    except Exception as e:
        out["err"] = str(e)[:200]
    return out


@app.get("/versions")
async def versions():
    """Report runtime versions + locate fully_shard for debugging."""
    out: dict = {}
    try:
        import torch  # type: ignore
        out["torch"] = torch.__version__
        out["torch.version.hip"] = getattr(torch.version, "hip", None)
        # Find where fully_shard actually lives in this torch build
        import torch.distributed.fsdp as fsdp
        out["fsdp_dir"] = sorted(x for x in dir(fsdp) if "shard" in x.lower())
        try:
            from torch.distributed._composable.fsdp import fully_shard
            out["composable_fsdp.fully_shard"] = "present"
        except Exception as e:
            out["composable_fsdp.fully_shard"] = f"MISSING: {e}"
        try:
            from torch.distributed.fsdp._fully_shard import fully_shard
            out["fsdp._fully_shard.fully_shard"] = "present"
        except Exception as e:
            out["fsdp._fully_shard.fully_shard"] = f"MISSING: {e}"
    except Exception as e:
        out["torch_error"] = str(e)
    try:
        import nemo  # type: ignore
        out["nemo"] = getattr(nemo, "__version__", "unknown")
    except Exception as e:
        out["nemo"] = f"MISSING: {e}"
    return out


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
    language: Optional[str] = Query(None, description="BCP-47; only Canary-1B-flash uses it today."),
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
            result = model_loader.transcribe(model, tmp_path, language=language)
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
