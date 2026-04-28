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
        samples, sr = sf.read(audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return self._embed_array(samples, sr)

    def _embed_array(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Embed a pre-loaded mono float32 numpy array. Used by
        verify_segments() to embed many windows from a single disk read."""
        import torch
        # pyannote needs at least ~16 ms of audio to produce a stable embedding;
        # silently zero-pad short windows.
        if len(samples) < int(0.016 * sr):
            samples = np.pad(samples, (0, int(0.016 * sr) - len(samples)))
        wav = torch.from_numpy(samples).unsqueeze(0)
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

    async def verify_segments(self, audio_path: str, *,
                              enrolled_embedding_b64: str,
                              config: dict) -> dict:
        """Per-segment user-tag via sliding window. Always returns the raw
        embedding per segment — schema is forward-compatible with multi-
        profile (v2) and auto-enroll cluster IDs (v3) without migration.
        """
        import soundfile as sf
        window_s = float(config.get("window_s", 1.0))
        hop_s = float(config.get("hop_s", 0.5))
        threshold = float(config.get("threshold", _DEFAULT_THRESHOLD))

        # One disk read; slice in memory for every window.
        samples, sr = sf.read(audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        duration_s = float(len(samples) / sr)

        enrolled = np.frombuffer(base64.b64decode(enrolled_embedding_b64),
                                 dtype=np.float32)
        enrolled_norm = float(np.linalg.norm(enrolled)) or 1.0

        win = int(window_s * sr)
        hop = max(1, int(hop_s * sr))
        segments: List[Dict[str, Any]] = []

        t0 = time.perf_counter()
        for start_idx in range(0, max(1, len(samples) - win + 1), hop):
            end_idx = min(start_idx + win, len(samples))
            chunk = samples[start_idx:end_idx]
            if len(chunk) < int(0.1 * sr):
                continue   # skip tail shorter than 100 ms
            emb = self._embed_array(chunk, sr)
            denom = float(np.linalg.norm(emb)) * enrolled_norm
            score = float(np.dot(enrolled, emb) / denom) if denom > 0 else 0.0
            segments.append({
                "start": float(start_idx / sr),
                "end": float(end_idx / sr),
                "embedding_b64": base64.b64encode(emb.tobytes()).decode("ascii"),
                "embedding_dim": int(emb.shape[0]),
                "score": score,
                "is_user": bool(score >= threshold),
            })
        wall_s = time.perf_counter() - t0

        return {
            "segments": segments,
            "n_segments": len(segments),
            "window_s": window_s,
            "hop_s": hop_s,
            "threshold": threshold,
            "duration_s": duration_s,
            "wall_time_s": wall_s,
            "cost_usd": 0.0,
        }
