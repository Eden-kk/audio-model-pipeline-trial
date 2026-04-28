"""Clip manifest CRUD — local JSON files under data/clips/<id>/.

Layout:
    data/clips/<clip_id>/manifest.json   — Clip metadata
    data/clips/<clip_id>/audio.<ext>     — Canonical audio bytes used by adapters
    data/clips/<clip_id>/source.<orig>   — (only for video uploads)
                                           Original .mp4/.mov/.webm kept for
                                           reference; adapters always read
                                           audio.wav (extracted via ffmpeg).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# Container formats whose audio track we extract via ffmpeg.
VIDEO_EXTENSIONS = {"mp4", "mov", "webm", "mkv", "avi", "m4v", "ts", "mts"}


def _data_root() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def _clip_dir(clip_id: str) -> Path:
    return _data_root() / "clips" / clip_id


@dataclass
class Clip:
    id: str
    source: str                          # "upload" | "record"
    modality: str = "audio"              # "audio" | "video" (video = audio extracted from a video container)
    filename: str = ""                   # canonical audio filename ("audio.wav" after extraction)
    format: str = ""                     # canonical format ("wav" | "mp3" | "opus" | …)
    original_filename: str = ""          # the upload's original name (e.g. "movie.mp4")
    original_format: str = ""            # original ext if different from canonical (mp4 / mov / …)
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
    # Plan D Stage A2 — populated for clips captured from /ws/mic with
    # ?save=1, where the vendor's streaming transcript is saved alongside
    # the audio as a ground-truth seed. None for upload/record clips.
    captured_transcript: Optional[str] = None
    captured_transcript_segments: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Clip":
        """Tolerant constructor for old manifests that pre-date later fields.

        Drops keys the current dataclass doesn't know about so a future
        downgrade can co-exist with manifests written by a newer version,
        and back-fills missing keys with defaults so manifests written
        before A2 still load. Without this, `Clip(**data)` would raise
        TypeError on either drift direction.
        """
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ─── ffmpeg audio extraction ─────────────────────────────────────────────────

class FFmpegMissingError(RuntimeError):
    """Raised when a video upload arrives but ffmpeg isn't on PATH."""


def _ffmpeg_bin() -> str:
    bin_path = shutil.which("ffmpeg")
    if not bin_path:
        raise FFmpegMissingError(
            "ffmpeg binary not found on PATH — required to extract audio from "
            "video uploads. Install via `brew install ffmpeg` (macOS) or "
            "`apt-get install ffmpeg` (Linux). The Docker image already includes it."
        )
    return bin_path


def extract_audio_track(source_path: Path, dest_wav: Path,
                        sample_rate: int = 16000, channels: int = 1) -> None:
    """Extract the audio track of a video container into WAV PCM s16le.

    Defaults match what most ASR adapters want (16 kHz mono).  Raises on
    failure so the upload route can return a clean 400/500 with detail.
    """
    cmd = [
        _ffmpeg_bin(),
        "-y", "-loglevel", "error",
        "-i", str(source_path),
        "-vn",                          # drop video
        "-ac", str(channels),           # mono
        "-ar", str(sample_rate),        # 16 kHz
        "-f", "wav", str(dest_wav),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode("utf-8", "replace")[:500]
        raise RuntimeError(f"ffmpeg failed: {msg}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out on {source_path.name}") from e


# ─── CRUD helpers ─────────────────────────────────────────────────────────────

def save_clip(clip: Clip, audio_bytes: bytes, extension: str) -> Clip:
    """Persist clip audio + manifest; returns the (possibly updated) clip.

    Video container handling: when `extension` is one of mp4/mov/webm/etc, the
    raw bytes are saved as `source.<ext>` and the audio track is extracted to
    `audio.wav` (16 kHz mono PCM).  Adapters always read `audio.wav`; the
    original is kept adjacent so the user can re-extract at higher quality
    later if they want.
    """
    d = _clip_dir(clip.id)
    d.mkdir(parents=True, exist_ok=True)

    extension = extension.lower().lstrip(".") or "wav"
    is_video = extension in VIDEO_EXTENSIONS

    if is_video:
        # Save the source video for reference, then strip out audio.wav.
        source_path = d / f"source.{extension}"
        source_path.write_bytes(audio_bytes)

        audio_path = d / "audio.wav"
        extract_audio_track(source_path, audio_path)

        clip.modality = "video"
        clip.original_format = extension
        clip.format = "wav"
    else:
        audio_path = d / f"audio.{extension}"
        audio_path.write_bytes(audio_bytes)
        clip.format = extension
        if not clip.original_format:
            clip.original_format = extension

    clip.filename = audio_path.name
    if not clip.original_filename:
        clip.original_filename = clip.filename

    (d / "manifest.json").write_text(
        json.dumps(clip.to_dict(), indent=2), encoding="utf-8"
    )
    return clip


def get_clip(clip_id: str) -> Optional[Clip]:
    manifest = _clip_dir(clip_id) / "manifest.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return Clip.from_dict(data)


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
                clips.append(Clip.from_dict(data))
            except Exception:
                pass
    return clips


def audio_path(clip_id: str) -> Optional[Path]:
    """Return the canonical audio file for clip_id (audio.* in the clip dir).

    Prefers `audio.wav` if both that and a `source.<ext>` exist (video uploads),
    so adapters always get the extracted audio rather than the raw container.
    """
    d = _clip_dir(clip_id)
    if not d.exists():
        return None
    # Prefer audio.wav explicitly
    wav = d / "audio.wav"
    if wav.exists():
        return wav
    for f in d.iterdir():
        if f.stem == "audio":
            return f
    return None


def source_path(clip_id: str) -> Optional[Path]:
    """Return the original-upload path (source.<ext>) for video clips, else None."""
    d = _clip_dir(clip_id)
    if not d.exists():
        return None
    for f in d.iterdir():
        if f.stem == "source":
            return f
    return None


def new_clip_id() -> str:
    return uuid.uuid4().hex
