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
    # Slow-loop additions (Slice 9.1)
    "language",         # output of LID adapters: {language, confidence}
    "memory_doc",       # intent_llm output envelope
    "dispatch_status",  # dispatch sink output (ack)
    # Realtime omni additions (Slice O2)
    "media_stream",     # multiplexed audio + (later) JPEG frames in
    "omni_event",       # adapter -> client: {audio_b64?, text_delta?, tool_call?, transcript?, done?}
}

ModelCategory = str
# Known categories:
#   "vad" | "asr" | "speaker_verify" | "diarization" | "tts" |
#   "intent_llm" |
#   "lid"             — language identification (Slice 9.1)
#   "dispatch"        — terminal sink (HaoClaw outbox / chat.send)
#   "realtime_omni"   — bidirectional audio (+video) ↔ audio+text+tool_calls
#                       single-stage fast-loop adapter. See omni_session().


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

    # --- streaming capability ---
    # True  → adapter implements transcribe_stream() (async generator that
    #         yields {partial_text, is_final, words?, raw?} dicts).
    #         The runner emits stage.progress events per yield, then a final
    #         stage.finished with the full text.
    # False → adapter only implements synchronous transcribe()/synthesize();
    #         the runner emits a single stage.finished with the full result.
    is_streaming: bool = False

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

    # ── Slow-loop additions (Slice 9.1) ─────────────────────────────────

    async def lid(self, audio_path: str, config: dict) -> dict:
        """LID adapters: detect spoken language in a clip prefix.
        Returns: {language: 'en'|'zh'|..., confidence: 0..1, candidates?: [...]}
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement lid(); "
            f"check adapter.category before dispatch."
        )

    async def verify_segments(
        self,
        audio_path: str,
        *,
        enrolled_embedding_b64: str,
        config: dict,
    ) -> dict:
        """Speaker-verify adapters: per-segment user/non-user labels via
        sliding window. ALWAYS returns the per-segment embedding so a
        future v2 (multi-profile) and v3 (cluster-based auto-enroll) can
        consume it without schema changes.

        Config knobs:
          window_s   default 1.0  — segment width in seconds
          hop_s      default 0.5  — stride between segments
          threshold  default 0.5  — match threshold for is_user

        Returns:
          {
            "segments": [
              {"start": float, "end": float, "embedding_b64": str,
               "score": float, "is_user": bool}
            ],
            "window_s": float, "hop_s": float, "threshold": float,
            "n_segments": int, "duration_s": float
          }
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement verify_segments(); "
            f"check adapter.category before dispatch."
        )

    async def infer(self, payload: dict, config: dict) -> dict:
        """intent_llm adapters: text + speaker_segments → structured envelope.

        Input payload shape:
          {"text": str,
           "words": [{"word", "start", "end", "speaker"?}],
           "speaker_segments": [{"start", "end", "is_user", "score"}],
           "language": str}

        Returns the slow-loop envelope:
          {"memory_doc": str,
           "tool_calls": [{"name", "args"}, ...],
           "salient_facts": [str, ...],
           "input_tokens": int, "output_tokens": int, "cost_usd": float}
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement infer(); "
            f"check adapter.category before dispatch."
        )

    async def dispatch(self, envelope: dict, config: dict) -> dict:
        """dispatch adapters: send the slow-loop envelope to a sink.

        Returns: {"sink": str, "ack": str, "bytes_written"?: int}
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement dispatch(); "
            f"check adapter.category before dispatch."
        )

    # ── Realtime omni (Slice O2) ────────────────────────────────────────

    async def omni_session(
        self,
        media_iter,          # AsyncIterator[dict]: each {"type":"audio"|"video"|"flush", "payload": bytes, "ts_ms": int}
        *,
        config: dict,
        context_iter=None,   # Optional[AsyncIterator[str]]: wearer-tag heartbeat lines, etc.
        abort_event=None,    # Optional[asyncio.Event]: when set, drop in-flight model response (interrupt support)
    ):
        """realtime_omni adapters: a bidirectional session.

        The runner (or /ws/omni proxy) hands the adapter:
          - `media_iter`: async iter of inbound media frames from the
            browser. Each frame is `{type, payload, ts_ms}`.
              type='audio'  → 16 kHz mono Int16 PCM bytes
              type='video'  → JPEG bytes (one frame)
              type='flush'  → end-of-utterance marker (push-to-talk release)
          - `context_iter` (optional): async iter of plain-text lines that
            should be appended to the system context as out-of-band signal.
            Used by the wearer-tag heartbeat to push "wearer just said: …"
            spans into the omni's prompt without interrupting media flow.
          - `config`: per-stage knobs (model name override, temperature,
            generate_audio bool, max_new_tokens, …).

        Yields adapter-side events as they arrive:
          {"type": "transcript",  "text": str, "is_final": bool}     # caption stream
          {"type": "text_delta",  "text": str}                       # response text token(s)
          {"type": "audio_b64",   "data": str, "sample_rate": int}   # response audio chunk
          {"type": "tool_call",   "name": str, "args": dict}         # function-call surface
          {"type": "done",        "latency_ms": float, "cost_usd": float}

        v1 implementations may collapse this into per-utterance HTTP calls
        (chunked-HTTP fallback) where each `flush` triggers a synchronous
        request; later v1.5 implementations can use a real WS to the model
        backend for sub-second TTFA.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement omni_session(); "
            f"check adapter.category before dispatch."
        )
        if False:
            yield  # type: ignore[unreachable]  (makes this an async generator)

    async def transcribe_stream(self, audio_path: str, config: dict):
        """ASR adapters with is_streaming=True: yield incremental partials.

        Each yielded dict has at minimum:
          {"partial_text": str, "is_final": bool}

        Optional extra keys:
          "words": [...]  (final only — full word-timing list)
          "language": "en"
          "raw": <vendor-specific payload>

        The last yield should have is_final=True with the full transcript;
        the runner uses that to write the run record.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement transcribe_stream(); "
            f"is_streaming should be False."
        )
        if False:
            yield  # type: ignore[unreachable]  (makes this an async generator)


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
                "is_streaming": bool(getattr(a, "is_streaming", False)),
            })
        return out


# Singleton registry — import this from everywhere.
registry = Registry()
