"""Resemblyzer speaker-verification adapter (CPU, in-process, no API key).

256-dim ECAPA-TDNN-style embedding, cosine similarity, threshold-based match.
Default threshold 0.60 (SNR-adaptive: 0.73 quiet / 0.60 noisy from B.1 sweep
in ambient-deploy/benchmarks/results/resemblyzer_threshold_sweep.csv).
"""
from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Optional

import numpy as np

_DEFAULT_THRESHOLD = 0.60


class ResemblyzerAdapter:
    id = "resemblyzer"
    category = "speaker_verify"
    display_name = "Resemblyzer (on-device)"
    hosting = "edge"
    vendor = "Resemblyzer (OSS)"

    inputs: List[Dict[str, str]] = [
        {"name": "audio", "type": "audio_file"},
        {"name": "enrolled_embedding", "type": "embedding"},
    ]
    outputs: List[Dict[str, str]] = [
        {"name": "score", "type": "score"},
    ]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "threshold": {"type": "number", "default": _DEFAULT_THRESHOLD,
                          "description": "Cosine threshold; 0.73 in quiet, 0.60 in noisy."},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = 0.0

    def __init__(self) -> None:
        self._encoder = None  # lazy

    def _load(self):
        if self._encoder is None:
            from resemblyzer import VoiceEncoder  # noqa
            self._encoder = VoiceEncoder()
        return self._encoder

    def _embed(self, audio_path: str) -> np.ndarray:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio_path)
        return self._load().embed_utterance(wav).astype(np.float32)

    async def enroll(self, audio_path: str, config: dict) -> dict:
        t0 = time.perf_counter()
        emb = self._embed(audio_path)
        wall_s = time.perf_counter() - t0
        return {
            "embedding_b64": base64.b64encode(emb.tobytes()).decode("ascii"),
            "embedding_dim": int(emb.shape[0]),
            "embedding_dtype": "float32",
            "wall_time_s": wall_s,
        }

    async def verify(self, audio_path: str, *, enrolled_embedding_b64: str,
                     config: dict) -> dict:
        threshold = float(config.get("threshold", _DEFAULT_THRESHOLD))
        enrolled = np.frombuffer(base64.b64decode(enrolled_embedding_b64),
                                 dtype=np.float32)

        t0 = time.perf_counter()
        test = self._embed(audio_path)
        wall_s = time.perf_counter() - t0

        # Cosine similarity
        denom = float(np.linalg.norm(enrolled) * np.linalg.norm(test))
        score = float(np.dot(enrolled, test) / denom) if denom > 0 else 0.0

        return {
            "score": score,
            "threshold": threshold,
            "match": bool(score >= threshold),
            "wall_time_s": wall_s,
            "cost_usd": 0.0,
        }
