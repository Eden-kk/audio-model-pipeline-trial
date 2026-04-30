"""NVIDIA Canary-Qwen-2.5B adapter — Canary encoder + Qwen-2.5 decoder.

Newer (2025) NeMo speech-LLM hybrid; stronger long-form WER than the
plain Canary-1B-flash.  EN-only.  Self-host via the AMD-ROCm model-server.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._nemo_http import (
    NEMO_INPUTS, NEMO_OUTPUTS, nemo_config_schema,
    transcribe_via_model_server,
)


class CanaryQwen25BAdapter:
    id = "canary_qwen_25b"
    category = "asr"
    display_name = "NVIDIA Canary-Qwen-2.5B (self-host)"
    hosting = "modal"
    vendor = "NVIDIA NeMo"

    inputs: List[Dict[str, str]] = NEMO_INPUTS
    outputs: List[Dict[str, str]] = NEMO_OUTPUTS
    config_schema: Dict[str, Any] = nemo_config_schema(default_lang="en")
    cost_per_call_estimate_usd: Optional[float] = 0.0
    is_streaming = True   # via chunked pseudo-stream
    supported_languages: List[str] = ["en"]
    multilang_realtime = False  # EN-only model

    async def transcribe(self, audio_path: str, config: dict) -> dict:
        return await transcribe_via_model_server(
            audio_path, model="canary-qwen-2.5b", language=config.get("language"),
        )

    async def transcribe_stream(self, audio_path: str, config: dict):
        from ._pseudo_stream import pseudo_stream_chunks
        async for ev in pseudo_stream_chunks(self, audio_path, config):
            yield ev
