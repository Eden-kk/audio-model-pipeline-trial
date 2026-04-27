"""Run persistence — append-only JSONL files under data/runs/<adapter>.jsonl.

Each line is one Run JSON object.  Reads scan the whole file; this is fine
for Slice 0 volumes (hundreds of runs).  Upgrade to SQLite in a later slice
if needed.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _data_root() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def _runs_file(adapter_id: str) -> Path:
    p = _data_root() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{adapter_id}.jsonl"


@dataclass
class Run:
    id: str
    clip_id: str
    adapter: str
    config: Dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: Optional[str] = None
    # Single-stage timing / cost
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    # Stage I/O
    input_preview: str = ""
    output_preview: str = ""
    raw_response: Any = None
    # Final transcript / result summary
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ─── CRUD helpers ─────────────────────────────────────────────────────────────

def append_run(run: Run) -> None:
    """Append one run record to the adapter's JSONL file."""
    path = _runs_file(run.adapter)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run.to_dict()) + "\n")


def get_run(run_id: str) -> Optional[Run]:
    """Scan all JSONL files for a run with this id."""
    runs_root = _data_root() / "runs"
    if not runs_root.exists():
        return None
    for jsonl in runs_root.glob("*.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("id") == run_id:
                    return Run(**data)
            except Exception:
                pass
    return None


def list_runs(adapter_id: Optional[str] = None) -> list:
    """List runs, optionally filtered by adapter_id."""
    runs_root = _data_root() / "runs"
    if not runs_root.exists():
        return []
    results = []
    pattern = f"{adapter_id}.jsonl" if adapter_id else "*.jsonl"
    for jsonl in sorted(runs_root.glob(pattern)):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(Run(**data))
            except Exception:
                pass
    return results


def new_run_id() -> str:
    return uuid.uuid4().hex
