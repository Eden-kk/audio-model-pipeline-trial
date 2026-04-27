"""Adapter protocol and Registry for the audio-trial backend.

Each adapter declares a category, typed input/output ports, and an async
transcribe() method (for ASR adapters).  The Registry is the single source of
truth for what adapters are available — /api/adapters reads straight from it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ─── Port types ──────────────────────────────────────────────────────────────

PortType = str  # one of the literals below; kept as str for JSON serialisation
PORT_TYPES = {
    "audio_file", "audio_stream", "audio_pcm", "text",
    "word_timing", "speech_segments", "speaker_segments",
    "embedding", "score", "tool_calls",
}

ModelCategory = str  # "vad" | "asr" | "speaker_verify" | "diarization" | "tts" | ...


# ─── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class Adapter(Protocol):
    """Structural protocol every adapter must satisfy."""

    # --- identity ---
    id: str
    category: ModelCategory          # "asr" | "tts" | "speaker_verify" | ...
    display_name: str
    hosting: str                     # "cloud" | "modal" | "edge"
    vendor: str

    # --- port declarations ---
    inputs: List[Dict[str, str]]     # [{"name": "audio", "type": "audio_file"}, ...]
    outputs: List[Dict[str, str]]    # [{"name": "text", "type": "text"}, ...]

    # --- schema / cost ---
    config_schema: Dict[str, Any]    # JSON Schema object describing run-time config knobs
    cost_per_call_estimate_usd: Optional[float]

    # --- main entry points (one per category; adapters implement the relevant ones) ---
    async def transcribe(self, audio_path: str, config: dict) -> dict:
        """ASR adapters: audio file → text + word-level timing.
        Returns at minimum: {text, words, language, duration_s, cost_usd}
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement transcribe(); "
            f"check adapter.category before dispatch."
        )

    async def synthesize(self, text: str, config: dict) -> dict:
        """TTS adapters: text → audio stream.
        Returns at minimum: {audio_bytes, mime, sample_rate, duration_s,
                              first_byte_ms, full_render_ms, cost_usd}
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement synthesize(); "
            f"check adapter.category before dispatch."
        )

    async def enroll(self, audio_path: str, config: dict) -> dict:
        """Speaker-verify adapters: enrol a reference clip → embedding bytes.
        Returns: {embedding_b64, embedding_dim, embedding_dtype, duration_s}
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement enroll(); "
            f"check adapter.category before dispatch."
        )

    async def verify(self, audio_path: str, *, enrolled_embedding_b64: str,
                     config: dict) -> dict:
        """Speaker-verify adapters: score a test clip against a stored embedding.
        Returns: {score, threshold, match, duration_s, cost_usd}
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement verify(); "
            f"check adapter.category before dispatch."
        )


# ─── Registry ─────────────────────────────────────────────────────────────────

class Registry:
    """In-process registry of Adapter instances keyed by adapter id."""

    def __init__(self) -> None:
        self._adapters: Dict[str, Any] = {}

    def register(self, adapter: Any) -> None:
        """Register an adapter instance.  The adapter must expose an `.id` attribute."""
        self._adapters[adapter.id] = adapter

    def get(self, adapter_id: str) -> Any:
        """Return the adapter instance or raise KeyError."""
        return self._adapters[adapter_id]

    def all(self) -> List[Any]:
        """Return all registered adapters in insertion order."""
        return list(self._adapters.values())

    def to_json(self) -> List[dict]:
        """Serialise the registry to a list of adapter metadata dicts
        (omitting the callable method — just the identity/schema fields)."""
        out = []
        for a in self._adapters.values():
            out.append({
                "id": a.id,
                "category": a.category,
                "display_name": a.display_name,
                "hosting": a.hosting,
                "vendor": a.vendor,
                "inputs": a.inputs,
                "outputs": a.outputs,
                "config_schema": a.config_schema,
                "cost_per_call_estimate_usd": a.cost_per_call_estimate_usd,
            })
        return out


# Singleton registry — import this from everywhere.
registry = Registry()
