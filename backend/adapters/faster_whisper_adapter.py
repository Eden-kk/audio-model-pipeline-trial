"""faster-whisper ASR adapter — multi-model variant.

Slice 1 update: the adapter now exposes a `model_name` config knob so a
single registered adapter can serve any of these checkpoints:
  - small.en             (Slice 0 default; ~75 MB int8; CPU-friendly)
  - large-v3             (Whisper Large v3; multilingual, 99 langs)
  - large-v3-turbo       (4× faster than v3 with negligible quality loss)
  - distil-large-v3      (6× faster via distillation, slight quality dip)
  - large-v3-turbo-en    (English-only turbo variant)

Each model is lazy-loaded once and cached in an in-process dict keyed by
(model_name, device, compute_type), so switching variants between calls is
free after the first warm-up.

No API key needed — pure local CPU/GPU inference.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

_MODELS: Dict[Tuple[str, str, str], Any] = {}
_MODEL_LOCK = threading.Lock()

# Default + supported model names (order matters in the enum below)
DEFAULT_MODEL_NAME = "small.en"
SUPPORTED_MODEL_NAMES = [
    "small.en",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
    "large-v3-turbo-en",
]
DEVICE = "cpu"          # use "cuda" or "rocm" if running on GPU
COMPUTE_TYPE = "int8"   # quantised for CPU; switch to "float16" on GPU


def _get_model(model_name: str, device: str = DEVICE,
               compute_type: str = COMPUTE_TYPE):
    """Return a WhisperModel singleton keyed by (model_name, device, compute_type)."""
    key = (model_name, device, compute_type)
    if key in _MODELS:
        return _MODELS[key]
    with _MODEL_LOCK:
        if key in _MODELS:
            return _MODELS[key]
        from faster_whisper import WhisperModel  # type: ignore[import]
        _MODELS[key] = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
    return _MODELS[key]


class FasterWhisperAdapter:
    id = "faster_whisper"
    category = "asr"
    display_name = "faster-whisper"
    hosting = "edge"
    vendor = "SYSTRAN"

    inputs: List[Dict[str, str]] = [
        {"name": "audio", "type": "audio_file"},
    ]
    outputs: List[Dict[str, str]] = [
        {"name": "text", "type": "text"},
        {"name": "words", "type": "word_timing"},
    ]

    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "default": DEFAULT_MODEL_NAME,
                "enum": SUPPORTED_MODEL_NAMES,
                "description": (
                    "Whisper checkpoint. small.en is small and CPU-friendly; "
                    "large-v3 is multilingual; large-v3-turbo is 4× faster than "
                    "v3 with little quality loss; distil-large-v3 is 6× faster."
                ),
            },
            "language": {
                "type": "string",
                "default": "en",
                "description": "ISO-639-1 code or 'auto' for auto-detect.",
            },
            "beam_size": {
                "type": "integer",
                "default": 5,
                "description": "Beam search width — higher is more accurate, slower.",
            },
            "word_timestamps": {
                "type": "boolean",
                "default": True,
            },
            "device": {
                "type": "string",
                "default": DEVICE,
                "enum": ["cpu", "cuda", "rocm"],
                "description": "Inference device. Use 'rocm' on the AMD remote machine.",
            },
            "compute_type": {
                "type": "string",
                "default": COMPUTE_TYPE,
                "enum": ["int8", "int8_float16", "float16", "float32"],
            },
        },
    }
    cost_per_call_estimate_usd: Optional[float] = 0.0

    async def transcribe(self, audio_path: str, config: dict) -> dict:
        model_name = config.get("model_name", DEFAULT_MODEL_NAME)
        if model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. "
                f"Allowed: {SUPPORTED_MODEL_NAMES}"
            )
        language_in = config.get("language", "en")
        language = None if language_in in ("auto", "", None) else language_in
        beam_size = int(config.get("beam_size", 5))
        word_timestamps = bool(config.get("word_timestamps", True))
        device = config.get("device", DEVICE)
        compute_type = config.get("compute_type", COMPUTE_TYPE)

        t0 = time.perf_counter()
        model = _get_model(model_name, device=device, compute_type=compute_type)
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
        )
        segments = list(segments_iter)
        wall_s = time.perf_counter() - t0

        words: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        for seg in segments:
            full_text_parts.append(seg.text.strip())
            if word_timestamps and seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": float(w.start),
                        "end": float(w.end),
                        "confidence": float(w.probability) if w.probability is not None else None,
                        "speaker": None,
                    })

        text = " ".join(full_text_parts)
        duration_s = float(info.duration) if info.duration else 0.0
        detected_language = info.language or (language_in or "en")

        return {
            "text": text,
            "words": words,
            "language": detected_language,
            "duration_s": duration_s,
            "cost_usd": 0.0,
            "wall_time_s": wall_s,
            "raw_response": {
                "model_name": model_name,
                "device": device,
                "compute_type": compute_type,
                "language_probability": (
                    float(info.language_probability)
                    if info.language_probability else None
                ),
                "duration": duration_s,
            },
        }
