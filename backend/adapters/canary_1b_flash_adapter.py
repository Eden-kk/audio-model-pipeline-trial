"""NVIDIA Canary-1B-flash adapter — multilingual EN/DE/ES/FR ASR.

Self-host via the AMD-ROCm model-server.  SOTA on Open ASR Leaderboard.
Wire-compatible with the Parakeet adapter (shared _nemo_http helper).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._nemo_http import (
    NEMO_INPUTS, NEMO_OUTPUTS, transcribe_via_model_server,
)


class Canary1BFlashAdapter:
    id = "canary_1b_flash"
    category = "asr"
    display_name = "NVIDIA Canary-1B-flash (self-host)"
    hosting = "modal"
    vendor = "NVIDIA NeMo"

    inputs: List[Dict[str, str]] = NEMO_INPUTS
    outputs: List[Dict[str, str]] = NEMO_OUTPUTS
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "language": {"type": "string", "default": "en",
                         "enum": ["en", "de", "es", "fr"],
                         "description": "Canary-1B-flash supports EN, DE, ES, FR."},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = 0.0

    async def transcribe(self, audio_path: str, config: dict) -> dict:
        return await transcribe_via_model_server(
            audio_path, model="canary-1b-flash"
        )
