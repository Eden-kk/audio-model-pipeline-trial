"""Lazy model loader for the trial-app model-server.

Each model has a load() function that returns a callable
`transcribe(audio_path) -> dict`.  Models are loaded on first request and
cached forever in this process.

Slice 1B status:
  - parakeet-tdt-1.1b: loader implemented (validated path)
  - canary-1b-flash:    loader stubbed; raises NotImplementedError until
                        Slice 2 loader-test on the AMD machine
  - canary-qwen-2.5b:   same — Slice 2 follow-up

Why stub Canary now?  We don't have an AMD machine in front of us during
authoring, so we can't validate the NeMo loader code path end-to-end.
Stubbing keeps the trial-app's adapters wired and lets the user spin up
the stack today; Slice 2 fills in the loaders once a real AMD GPU is
available for testing.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

_LOADED: Dict[str, Tuple[Any, str]] = {}  # model_id → (instance, version_string)
_LOAD_LOCK = threading.Lock()


# ── Parakeet-TDT-1.1B ────────────────────────────────────────────────────────

def _load_parakeet() -> Tuple[Any, str]:
    from nemo.collections.asr.models import ASRModel  # type: ignore[import]
    model = ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
    model.eval()
    return model, "nvidia/parakeet-tdt-1.1b"


def _transcribe_parakeet(model: Any, audio_path: str) -> Dict[str, Any]:
    out = model.transcribe([audio_path])
    # NeMo's transcribe shape varies by version; normalise:
    hyp = out
    while isinstance(hyp, list) and hyp:
        hyp = hyp[0]
    text = getattr(hyp, "text", None) or (hyp if isinstance(hyp, str) else "")
    words: list = []
    if hasattr(hyp, "words") and hyp.words:
        for w in hyp.words:
            words.append({
                "word": getattr(w, "word", ""),
                "start": float(getattr(w, "start", 0.0) or 0.0),
                "end": float(getattr(w, "end", 0.0) or 0.0),
            })
    return {"text": text, "words": words, "language": "en"}


# ── Canary-1B-flash (stubbed for Slice 1B) ──────────────────────────────────

def _load_canary_1b_flash() -> Tuple[Any, str]:
    raise NotImplementedError(
        "canary-1b-flash loader pending Slice 2 — needs validation on a real "
        "AMD-ROCm GPU. Slice 1B intentionally stubs this so the rest of the "
        "stack can ship; the trial-app's adapter wires through to a clear "
        "503 'model not loaded' error in the meantime."
    )


# ── Canary-Qwen-2.5B (stubbed for Slice 1B) ─────────────────────────────────

def _load_canary_qwen_25b() -> Tuple[Any, str]:
    raise NotImplementedError(
        "canary-qwen-2.5b loader pending Slice 2 — same as canary-1b-flash."
    )


# ── Registry ─────────────────────────────────────────────────────────────────

LOADERS: Dict[str, Callable[[], Tuple[Any, str]]] = {
    "parakeet-tdt-1.1b": _load_parakeet,
    "canary-1b-flash": _load_canary_1b_flash,
    "canary-qwen-2.5b": _load_canary_qwen_25b,
}

TRANSCRIBERS: Dict[str, Callable[[Any, str], Dict[str, Any]]] = {
    "parakeet-tdt-1.1b": _transcribe_parakeet,
    # Canary transcribers will land in Slice 2 alongside their loaders.
}


def status() -> Dict[str, Any]:
    """Return the load state of every registered model."""
    out = {}
    for mid in LOADERS:
        loaded = mid in _LOADED
        out[mid] = {
            "loaded": loaded,
            "version": _LOADED[mid][1] if loaded else None,
            "transcriber_implemented": mid in TRANSCRIBERS,
        }
    return out


def get_or_load(model_id: str) -> Any:
    """Return the model instance for `model_id`, loading on first call.

    Raises:
        KeyError if model_id is unknown.
        NotImplementedError if the loader is intentionally stubbed.
        Anything the actual loader raises (network, GPU, etc.).
    """
    if model_id not in LOADERS:
        raise KeyError(f"unknown model id: {model_id!r}")
    if model_id in _LOADED:
        return _LOADED[model_id][0]
    with _LOAD_LOCK:
        if model_id in _LOADED:
            return _LOADED[model_id][0]
        instance, version = LOADERS[model_id]()
        _LOADED[model_id] = (instance, version)
        return instance


def transcribe(model_id: str, audio_path: str) -> Dict[str, Any]:
    if model_id not in TRANSCRIBERS:
        raise NotImplementedError(
            f"transcriber for {model_id!r} not implemented yet — see model_loader.py"
        )
    instance = get_or_load(model_id)
    t0 = time.perf_counter()
    out = TRANSCRIBERS[model_id](instance, audio_path)
    out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
    out["model"] = model_id
    return out
