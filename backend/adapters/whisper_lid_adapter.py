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

        # faster-whisper >= 1.0 expects either a file path OR a numpy
        # array, but `detect_language(...)` has historically been
        # finicky about path inputs across versions. The robust path
        # is to call `model.transcribe(...)` with `language=None`,
        # which always returns info.language + info.language_probability
        # populated regardless of major-version shifts.
        t0 = time.perf_counter()
        # We don't need the transcript — just the side-effect language
        # detection. beam_size=1 keeps it fast.
        segments_iter, info = model.transcribe(
            audio_path,
            language=None,
            beam_size=1,
            best_of=1,
            without_timestamps=True,
            vad_filter=False,
        )
        # Consume the generator (lazy in faster-whisper) so info is
        # finalised. We discard the segments.
        for _ in segments_iter:
            break  # one segment is enough to populate info
        wall_s = time.perf_counter() - t0

        lang_code = info.language or "en"
        lang_prob = float(info.language_probability or 0.0)

        # detect_language() (when it does work) returns the dict of
        # all-language probs; transcribe() doesn't expose that. We
        # synthesize a single-candidate list to keep the output shape
        # consistent.
        candidates = [{"language": lang_code, "confidence": lang_prob}]

        # As a bonus second-pass, run detect_language() on the loaded
        # audio for the full top-K distribution. Defensive — if it
        # errors, we just keep the single-candidate list.
        try:
            import numpy as np
            import soundfile as sf
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # faster-whisper's detect_language wants 16 kHz; cheap resample.
            if sr != 16000:
                # linear-interpolate (good enough for LID)
                ratio = 16000.0 / sr
                new_n = int(audio.size * ratio)
                idx = np.minimum(
                    (np.arange(new_n) / ratio).astype(np.int64),
                    audio.size - 1,
                )
                audio = audio[idx].astype(np.float32)
            res = model.detect_language(audio[: 16000 * 30])
            if isinstance(res, tuple) and len(res) >= 3:
                _lang2, _prob2, all_probs = res
                if isinstance(all_probs, dict):
                    candidates = [
                        {"language": code, "confidence": float(prob)}
                        for code, prob in sorted(
                            all_probs.items(), key=lambda kv: -kv[1]
                        )[:top_k]
                    ]
                    if candidates:
                        lang_code = candidates[0]["language"]
                        lang_prob = candidates[0]["confidence"]
        except Exception:
            pass

        return {
            "language": lang_code,
            "confidence": lang_prob,
            "candidates": candidates,
            "wall_time_s": wall_s,
            "cost_usd": 0.0,
            "model_name": model_name,
        }
