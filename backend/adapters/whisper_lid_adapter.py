"""Whisper-based Language ID adapter.

Lightweight LID using faster-whisper-tiny — runs `model.detect_language()`
on the first ~30 s of audio, returns top language + confidence + top-K
candidates. Reuses the same WhisperModel cache as the faster-whisper ASR
adapter when both are loaded with matching model_name + device + compute.

No API key needed. Local CPU inference; ~200 ms on a short clip.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[tuple, Any] = {}

DEFAULT_MODEL_NAME = "tiny"   # tiny is plenty for LID (10× faster than small)
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"


def _get_model(model_name: str, device: str, compute_type: str):
    key = (model_name, device, compute_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        from faster_whisper import WhisperModel  # type: ignore[import]
        _MODEL_CACHE[key] = WhisperModel(model_name, device=device,
                                         compute_type=compute_type)
    return _MODEL_CACHE[key]


class WhisperLIDAdapter:
    id = "whisper_lid"
    category = "lid"
    display_name = "Whisper LID (tiny, faster-whisper)"
    hosting = "edge"
    vendor = "SYSTRAN"
    is_streaming = False

    inputs: List[Dict[str, str]] = [{"name": "audio", "type": "audio_file"}]
    outputs: List[Dict[str, str]] = [{"name": "language", "type": "language"}]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "default": DEFAULT_MODEL_NAME,
                           "enum": ["tiny", "base", "small"],
                           "description": "Whisper checkpoint for LID."},
            "device": {"type": "string", "default": DEFAULT_DEVICE,
                       "enum": ["cpu", "cuda", "rocm"]},
            "compute_type": {"type": "string", "default": DEFAULT_COMPUTE_TYPE,
                             "enum": ["int8", "int8_float16", "float16", "float32"]},
            "top_k": {"type": "integer", "default": 3,
                      "description": "Return top-K candidate languages."},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = 0.0

    async def lid(self, audio_path: str, config: dict) -> dict:
        model_name = config.get("model_name", DEFAULT_MODEL_NAME)
        device = config.get("device", DEFAULT_DEVICE)
        compute_type = config.get("compute_type", DEFAULT_COMPUTE_TYPE)
        top_k = int(config.get("top_k", 3))

        model = _get_model(model_name, device, compute_type)

        # faster-whisper exposes detect_language() that returns
        # (lang_code, lang_probability, dict_of_all_probs).
        t0 = time.perf_counter()
        try:
            lang_code, lang_prob, all_probs = model.detect_language(audio_path)
        except TypeError:
            # Older faster-whisper returns (lang, prob) only
            lang_code, lang_prob = model.detect_language(audio_path)
            all_probs = {lang_code: lang_prob}
        wall_s = time.perf_counter() - t0

        candidates = []
        if isinstance(all_probs, dict):
            for code, prob in sorted(all_probs.items(),
                                     key=lambda kv: -kv[1])[:top_k]:
                candidates.append({"language": code, "confidence": float(prob)})

        return {
            "language": lang_code,
            "confidence": float(lang_prob),
            "candidates": candidates,
            "wall_time_s": wall_s,
            "cost_usd": 0.0,
            "model_name": model_name,
        }
