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
import torchaudio  # noqa: F401

# ── torchaudio compatibility shim for pyannote.audio 3.x on torchaudio ≥ 2.9 ──
#
# torchaudio 2.9+ migrated audio I/O to torchcodec and removed three symbols
# that pyannote.audio 3.x still references:
#   • torchaudio.AudioMetaData  – return-type annotation on io.py:get_torchaudio_info()
#   • torchaudio.list_audio_backends – called in Audio.__init__ to pick a backend
#   • torchaudio.info            – called in get_torchaudio_info() for file-path inputs
#
# Our adapter bypasses torchaudio audio I/O entirely (soundfile → torch tensor →
# {"waveform", "sample_rate"} dict passed to Inference), so we never actually
# execute those code paths at runtime.  However pyannote.audio evaluates these
# attribute accesses at *module-import time* (type annotation + Audio.__init__
# called from protocol.py at module level), which means they must exist before
# `from pyannote.audio import Inference, Model` is called.
#
# Upstream fix: pyannote.audio 4.x rewrites I/O on top of torchcodec.  A major-
# version upgrade is deferred because 4.x pulls in opentelemetry and pyannoteai-sdk
# and may change embedding dimensions.  This shim is the minimum patch for 3.x.
#
# Invariant to enforce: all three guards use `not hasattr(torchaudio, ...)` so
# they are no-ops if a future torchaudio re-adds the symbols.
if not hasattr(torchaudio, "AudioMetaData"):
    class _AudioMetaDataShim:
        """Drop-in for torchaudio.AudioMetaData (removed in torchaudio 2.9)."""
        __slots__ = ("sample_rate", "num_frames", "num_channels",
                     "bits_per_sample", "encoding")

        def __init__(self, sample_rate: int = 0, num_frames: int = 0,
                     num_channels: int = 0, bits_per_sample: int = 0,
                     encoding: str = "") -> None:
            self.sample_rate = int(sample_rate)
            self.num_frames = int(num_frames)
            self.num_channels = int(num_channels)
            self.bits_per_sample = int(bits_per_sample)
            self.encoding = str(encoding)

    torchaudio.AudioMetaData = _AudioMetaDataShim  # type: ignore[attr-defined]

if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends() -> list:
        """Shim for torchaudio.list_audio_backends (removed in torchaudio 2.9).
        Returns ['soundfile'] so Audio.__init__ falls back to the soundfile backend."""
        return ["soundfile"]

    torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]

if not hasattr(torchaudio, "info"):
    def _info(path: str, backend: str = None) -> "torchaudio.AudioMetaData":
        """Shim for torchaudio.info (removed in torchaudio 2.9) via soundfile."""
        import soundfile as sf
        _sf_info = sf.info(path)
        return torchaudio.AudioMetaData(  # type: ignore[attr-defined]
            sample_rate=_sf_info.samplerate,
            num_frames=_sf_info.frames,
            num_channels=_sf_info.channels,
            bits_per_sample=0,
            encoding="PCM_S",
        )

    torchaudio.info = _info  # type: ignore[attr-defined]

# ── huggingface_hub compatibility shim for pyannote.audio 3.x on hf_hub ≥ 1.0 ──
#
# huggingface_hub 1.0 renamed `use_auth_token` → `token` in hf_hub_download().
# pyannote.audio 3.x's Model.from_pretrained() still passes `use_auth_token=` to
# hf_hub_download, which raises TypeError on newer hf_hub.
# Wrap hf_hub_download to silently forward use_auth_token as token.
import huggingface_hub as _hfhub

_orig_hf_hub_download = _hfhub.hf_hub_download


def _compat_hf_hub_download(*args: Any, use_auth_token: Optional[str] = None,
                            **kwargs: Any) -> str:
    """Shim: translate deprecated use_auth_token kwarg to token for hf_hub ≥ 1.0."""
    if use_auth_token is not None and "token" not in kwargs:
        kwargs["token"] = use_auth_token
    return _orig_hf_hub_download(*args, **kwargs)


_hfhub.hf_hub_download = _compat_hf_hub_download  # type: ignore[assignment]

# ── lightning_fabric pl_load shim for torch ≥ 2.6 safe-load default change ──
#
# torch 2.6 changed torch.load()'s default from weights_only=None/False to True.
# The pyannote/embedding .ckpt includes pytorch_lightning callback state (e.g.,
# EarlyStopping) that torch.load rejects under weights_only=True.
# lightning_fabric._load passes weights_only=None for local files, which now
# triggers the strict default.  Since pyannote/embedding is a trusted model
# (gated by HF_TOKEN + our own accepted license), forcing weights_only=False
# for local .ckpt paths is safe.
import lightning_fabric.utilities.cloud_io as _lcio

_orig_pl_load = _lcio._load


def _patched_pl_load(path_or_url: Any, map_location: Any = None,
                     weights_only: Optional[bool] = None) -> Any:
    """Shim: use weights_only=False for local .ckpt files (trusted pyannote checkpoint)."""
    if weights_only is None and not str(path_or_url).startswith("http"):
        weights_only = False
    return _orig_pl_load(path_or_url, map_location=map_location,
                         weights_only=weights_only)


_lcio._load = _patched_pl_load  # type: ignore[assignment]

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
            import pyannote.audio.core.model as _pam
            _pam.hf_hub_download = _compat_hf_hub_download  # patch in module namespace too
            model = Model.from_pretrained(_MODEL, use_auth_token=tok)
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
