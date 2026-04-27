"""pyannote/embedding speaker-verification adapter (CPU, in-process).

512-dim ECAPA-TDNN embedding via pyannote.audio.  Uses the soundfile-bypass
trick documented in ambient-deploy to dodge the torchcodec ROCm/macOS issue:
read audio with soundfile → torch tensor → pass dict directly to Inference.

Env: HF_TOKEN (license must be accepted at hf.co/pyannote/embedding).
Default threshold: 0.50 (matches the validated production gate).
"""
from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

_MODEL = "pyannote/embedding"
_DEFAULT_THRESHOLD = 0.50


class PyannoteVerifyAdapter:
    id = "pyannote_verify"
    category = "speaker_verify"
    display_name = "pyannote/embedding"
    hosting = "edge"  # CPU in-process; can also run as a Modal/AMD service
    vendor = "pyannote"

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
                          "description": "Cosine threshold (validated 0.5 from B.1)."},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = 0.0

    def __init__(self) -> None:
        self._inference = None

    def _load(self):
        if self._inference is None:
            tok = os.environ.get("HF_TOKEN", "")
            if not tok:
                raise RuntimeError(
                    "HF_TOKEN not set — needed for pyannote/embedding gated model. "
                    "Accept the license at https://huggingface.co/pyannote/embedding "
                    "then add HF_TOKEN to backend/.env."
                )
            from pyannote.audio import Inference, Model
            model = Model.from_pretrained(_MODEL, token=tok)
            self._inference = Inference(model, window="whole")
        return self._inference

    def _embed(self, audio_path: str) -> np.ndarray:
        import soundfile as sf
        import torch

        samples, sr = sf.read(audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        wav = torch.from_numpy(samples).unsqueeze(0)  # (1, T)
        emb = self._load()({"waveform": wav, "sample_rate": sr})
        return np.asarray(emb.data).astype(np.float32).reshape(-1)

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

        denom = float(np.linalg.norm(enrolled) * np.linalg.norm(test))
        score = float(np.dot(enrolled, test) / denom) if denom > 0 else 0.0

        return {
            "score": score,
            "threshold": threshold,
            "match": bool(score >= threshold),
            "wall_time_s": wall_s,
            "cost_usd": 0.0,
        }
