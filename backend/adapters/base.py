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

    # --- main entry point (ASR adapters) ---
    async def transcribe(self, audio_path: str, config: dict) -> dict:
        """Run ASR on the given audio file; return a dict with at minimum:
        {text, words, language, duration_s, cost_usd}
        """
        ...


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
