"""Parakeet-TDT-1.1B adapter — talks to the AMD-hosted model-server.

Self-host on the remote AMD-ROCm machine (Slice 1B's model-server container).
Wire shape: shared with Canary adapters via ._nemo_http.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._nemo_http import (
    NEMO_INPUTS, NEMO_OUTPUTS, nemo_config_schema,
    transcribe_via_model_server,
)


class ParakeetAdapter:
    id = "parakeet"
    category = "asr"
    display_name = "Parakeet-TDT-1.1B (self-host)"
    hosting = "modal"  # actually AMD-ROCm via model-server, but kept as "self-host" semantically
    vendor = "NVIDIA NeMo"

    inputs: List[Dict[str, str]] = NEMO_INPUTS
    outputs: List[Dict[str, str]] = NEMO_OUTPUTS
    config_schema: Dict[str, Any] = nemo_config_schema(default_lang="en")
    cost_per_call_estimate_usd: Optional[float] = 0.0

    async def transcribe(self, audio_path: str, config: dict) -> dict:
        return await transcribe_via_model_server(audio_path, model="parakeet-tdt-1.1b")
