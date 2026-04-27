"""Clip manifest CRUD — local JSON files under data/clips/<id>/.

Layout:
    data/clips/<clip_id>/manifest.json   — Clip metadata
    data/clips/<clip_id>/audio.<ext>     — Raw audio bytes
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


def _data_root() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def _clip_dir(clip_id: str) -> Path:
    return _data_root() / "clips" / clip_id


@dataclass
class Clip:
    id: str
    source: str                          # "upload" | "record"
    modality: str = "audio"
    filename: str = ""
    format: str = ""                     # "wav" | "mp3" | "opus" | …
    duration_s: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    language_detected: Optional[str] = None
    snr_db: Optional[float] = None
    speaker_count_estimate: Optional[int] = None
    user_tags: List[str] = field(default_factory=list)
    scenarios: List[str] = field(default_factory=list)
    uploaded_by: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─── CRUD helpers ─────────────────────────────────────────────────────────────

def save_clip(clip: Clip, audio_bytes: bytes, extension: str) -> Clip:
    """Persist clip audio + manifest; returns the (possibly updated) clip."""
    d = _clip_dir(clip.id)
    d.mkdir(parents=True, exist_ok=True)

    audio_path = d / f"audio.{extension}"
    audio_path.write_bytes(audio_bytes)

    clip.filename = audio_path.name
    clip.format = extension

    (d / "manifest.json").write_text(
        json.dumps(clip.to_dict(), indent=2), encoding="utf-8"
    )
    return clip


def get_clip(clip_id: str) -> Optional[Clip]:
    manifest = _clip_dir(clip_id) / "manifest.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return Clip(**data)


def list_clips() -> List[Clip]:
    clips_root = _data_root() / "clips"
    if not clips_root.exists():
        return []
    clips = []
    for d in sorted(clips_root.iterdir()):
        manifest = d / "manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                clips.append(Clip(**data))
            except Exception:
                pass
    return clips


def audio_path(clip_id: str) -> Optional[Path]:
    """Return the path to the audio file for clip_id, or None if missing."""
    d = _clip_dir(clip_id)
    if not d.exists():
        return None
    for f in d.iterdir():
        if f.stem == "audio":
            return f
    return None


def new_clip_id() -> str:
    return uuid.uuid4().hex
