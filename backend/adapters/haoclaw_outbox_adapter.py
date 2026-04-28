"""HaoClaw-outbox dispatch adapter — terminal sink for the slow-loop pipeline.

V1 behaviour: append the slow-loop envelope to a local JSONL file
(default `data/haoclaw_outbox.jsonl`). Each line is a self-contained
record the future real `chat.send` integration can replay.

When INTENT_LLM_URL-style HAOCLAW_URL becomes available, swap the impl to
POST the same envelope to that endpoint. Schema does not change.
"""
from __future__ import annotations

import datetime
import json as _json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _outbox_path() -> Path:
    base = Path(os.environ.get("DATA_DIR", "data"))
    return base / "haoclaw_outbox.jsonl"


class HaoClawOutboxAdapter:
    id = "haoclaw_outbox"
    category = "dispatch"
    display_name = "HaoClaw outbox (local file)"
    hosting = "edge"
    vendor = "HaoClaw"
    is_streaming = False

    inputs: List[Dict[str, str]] = [
        {"name": "envelope", "type": "memory_doc"},
    ]
    outputs: List[Dict[str, str]] = [
        {"name": "ack", "type": "dispatch_status"},
    ]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "session_key": {
                "type": "string", "default": "",
                "description": "Optional session key — defaults to ambient:memory:YYYY-MM-DD.",
            },
            "include_envelope_in_response": {
                "type": "boolean", "default": True,
            },
        },
    }
    cost_per_call_estimate_usd: Optional[float] = 0.0

    async def dispatch(self, envelope: dict, config: dict) -> dict:
        session_key = config.get("session_key") or _default_session_key()
        path = _outbox_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "session_key": session_key,
            "envelope": envelope,
        }
        line = _json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        out = {
            "sink": "local_file",
            "ack": f"appended to {path}",
            "session_key": session_key,
            "bytes_written": len(line) + 1,
            "outbox_path": str(path),
        }
        if config.get("include_envelope_in_response", True):
            out["envelope"] = envelope
        return out


def _default_session_key() -> str:
    today = datetime.datetime.utcnow().date().isoformat()
    return f"ambient:memory:{today}"
