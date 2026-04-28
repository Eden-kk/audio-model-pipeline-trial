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

    def _embed_array(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Embed a pre-loaded mono float32 numpy array. Used by
        verify_segments() — Resemblyzer's preprocess_wav() expects a
        float32 array at 16 kHz, so we resample if needed once per call."""
        from resemblyzer import preprocess_wav
        # preprocess_wav can take an ndarray when source_sr is provided
        try:
            wav = preprocess_wav(samples, source_sr=sr)
        except TypeError:
            # older API: just resample inline
            if sr != 16000:
                # poor-man's resample (drop into scipy when available)
                try:
                    import scipy.signal  # type: ignore
                    samples = scipy.signal.resample_poly(samples, 16000, sr).astype(np.float32)
                except Exception:
                    pass
            wav = samples
        return self._load().embed_utterance(wav).astype(np.float32)

    async def verify_segments(self, audio_path: str, *,
                              enrolled_embedding_b64: str,
                              config: dict) -> dict:
        """Per-segment user-tag via sliding window. Output schema mirrors
        pyannote_verify's verify_segments — same field names and shapes
        so the runner doesn't have to branch on adapter id."""
        import soundfile as sf
        window_s = float(config.get("window_s", 1.0))
        hop_s = float(config.get("hop_s", 0.5))
        threshold = float(config.get("threshold", _DEFAULT_THRESHOLD))

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
                continue
            try:
                emb = self._embed_array(chunk, sr)
            except Exception:
                # short windows can fail Resemblyzer's VAD-pad step; skip
                continue
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
