"""faster-whisper (small.en) ASR adapter for the audio-trial backend.

Lazy-loads WhisperModel on the first transcribe() call and reuses the
singleton for all subsequent calls (avoid 1-2 s model-load overhead per run).

No API key needed — runs fully locally on CPU with int8 quantisation.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

_MODEL_LOCK = threading.Lock()
_MODEL_INSTANCE = None  # type: ignore[assignment]

MODEL_SIZE = "small.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


def _get_model():
    """Return the WhisperModel singleton, loading it on first call."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        with _MODEL_LOCK:
            if _MODEL_INSTANCE is None:
                from faster_whisper import WhisperModel  # type: ignore[import]
                _MODEL_INSTANCE = WhisperModel(
                    MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE
                )
    return _MODEL_INSTANCE


class FasterWhisperAdapter:
    # ── Adapter identity ────────────────────────────────────────────────
    id = "faster_whisper"
    category = "asr"
    display_name = "faster-whisper (small.en)"
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
            "language": {
                "type": "string",
                "default": "en",
                "description": "ISO-639-1 language code or null for auto-detect.",
            },
            "beam_size": {
                "type": "integer",
                "default": 5,
                "description": "Beam search width — higher is more accurate but slower.",
            },
            "word_timestamps": {
                "type": "boolean",
                "default": True,
                "description": "Whether to return per-word timestamps.",
            },
        },
    }
    cost_per_call_estimate_usd: Optional[float] = 0.0  # local model, no API cost

    # ── Public transcribe ───────────────────────────────────────────────
    async def transcribe(self, audio_path: str, config: dict) -> dict:
        """Run ASR via faster-whisper and return a normalised result dict.

        Returns:
            {text, words, language, duration_s, cost_usd, raw_response}
        """
        language = config.get("language", "en") or None  # None means auto-detect
        beam_size = int(config.get("beam_size", 5))
        word_timestamps = bool(config.get("word_timestamps", True))

        t0 = time.perf_counter()
        model = _get_model()

        # faster_whisper.transcribe is synchronous — fine for Slice 0
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
        )
        segments = list(segments_iter)
        wall_s = time.perf_counter() - t0

        # Build flat word list from all segments
        words = []
        full_text_parts = []
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
        detected_language = info.language or (language or "en")

        return {
            "text": text,
            "words": words,
            "language": detected_language,
            "duration_s": duration_s,
            "cost_usd": 0.0,
            "wall_time_s": wall_s,
            "raw_response": {
                "model": MODEL_SIZE,
                "language_probability": float(info.language_probability)
                if info.language_probability else None,
                "duration": duration_s,
            },
        }
