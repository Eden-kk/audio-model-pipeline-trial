"""Curated benchmark-clip ingest for the trial-app corpus.

Pulls a small representative set of audio from
  ambient-deploy/benchmarks/test_clips/...
into the local trial-app via POST /api/clips, then PATCHes scenario +
user-tag chips so the Library page can filter by category.

Categories shipped (43 clips total — keep small enough to iterate fast):

  realtime-prompts        10  — q01..q10 (Kokoro-TTS assistant queries)
  english-utterances       5  — test_01..test_05 (longer English samples)
  multilingual             8  — 2 clips × {es, fr, ja, de} (pub_*_00, pub_*_01)
  verification-clean       6  — LibriSpeech same/diff pairs (clean)
  verification-noise-5db   6  — same pairs + MUSAN @ +5 dB SNR
  verification-noise-0db   6  — same pairs + MUSAN @  0 dB SNR (worst-case)
  scenario-multi-speaker   2  — interleaved BAB / ABABA patterns

Each clip is auto-tagged (no manual UI work required):
  - language_detected   from source-dir convention (en/zh/es/fr/ja/de)
  - snr_db              energy-based estimate from the WAV itself
  - speaker_count_estimate  from filename pattern (BAB=2, ABABA=2, else 1)

Each clip gets:
  scenarios=[primary_category, ...secondary_tags]
  user_tags=[provenance/source-info]

Usage:
  python3 scripts/ingest_benchmark_corpus.py [--api http://localhost:8000]
                                             [--purge-existing]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import httpx

# Allow importing the trial-app's adapters directly so we can run real LID
# without going through the HTTP layer. Assumes this script runs inside the
# same venv as the backend (backend/.venv).
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


BENCHMARK_ROOT = Path(
    "/Users/yvette/code/ambient-deploy/.claude/worktrees/"
    "heuristic-chebyshev/benchmarks/test_clips"
)


# ─── Curated clip selection ──────────────────────────────────────────────────
#
# Each entry: (relative_path, scenarios_list, user_tags_list, note)

REALTIME_PROMPTS: List[Tuple[str, List[str], List[str], str]] = [
    (f"realtime_user_audio/q{i:02d}.wav",
     ["realtime-prompts", "clean-en", "assistant-query", "tts-synth"],
     ["kokoro-tts", "ar-glass-style"],
     f"q{i:02d} — assistant-style user query (Kokoro-TTS)")
    for i in range(1, 11)
]

ENGLISH_UTTERANCES: List[Tuple[str, List[str], List[str], str]] = [
    ("en/test_01_team_standup.wav",
     ["english-utterances", "clean-en", "assistant-query", "tts-synth"],
     ["kokoro-tts", "long-form"],
     "team standup — multi-sentence English"),
    ("en/test_02_set_reminder.wav",
     ["english-utterances", "clean-en", "assistant-query", "tts-synth"],
     ["kokoro-tts"],
     "set-reminder intent"),
    ("en/test_03_api_deadline.wav",
     ["english-utterances", "clean-en", "assistant-query", "tts-synth"],
     ["kokoro-tts", "technical-vocab"],
     "API deadline — technical vocabulary"),
    ("en/test_04_casual_weather.wav",
     ["english-utterances", "clean-en", "assistant-query", "tts-synth"],
     ["kokoro-tts", "casual"],
     "casual weather chat"),
    ("en/test_05_critical_server.wav",
     ["english-utterances", "clean-en", "assistant-query", "tts-synth"],
     ["kokoro-tts", "technical-vocab"],
     "critical server alert — technical vocabulary"),
]

# 3 same-speaker positives + 3 different-speaker negatives → speaker verify
VERIFICATION_CLEAN: List[Tuple[str, List[str], List[str], str]] = [
    ("verification/2035_same_00.wav",
     ["verification-clean", "librispeech", "verification-positive", "clean-en"],
     ["librispeech-2035", "real-human-speech"],
     "speaker 2035 same-pair positive #00"),
    ("verification/1673_same_00.wav",
     ["verification-clean", "librispeech", "verification-positive", "clean-en"],
     ["librispeech-1673", "real-human-speech"],
     "speaker 1673 same-pair positive #00"),
    ("verification/84_same_00.wav",
     ["verification-clean", "librispeech", "verification-positive", "clean-en"],
     ["librispeech-84", "real-human-speech"],
     "speaker 84 same-pair positive #00"),
    ("verification/1673_diff_00_from_1988.wav",
     ["verification-clean", "librispeech", "verification-negative", "clean-en"],
     ["librispeech-1673", "librispeech-1988", "real-human-speech"],
     "diff-pair: enrolled 1673 vs probe 1988"),
    ("verification/84_diff_00_from_1673.wav",
     ["verification-clean", "librispeech", "verification-negative", "clean-en"],
     ["librispeech-84", "librispeech-1673", "real-human-speech"],
     "diff-pair: enrolled 84 vs probe 1673"),
    ("verification/2035_diff_00_from_1673.wav",
     ["verification-clean", "librispeech", "verification-negative", "clean-en"],
     ["librispeech-2035", "librispeech-1673", "real-human-speech"],
     "diff-pair: enrolled 2035 vs probe 1673"),
]

VERIFICATION_NOISE_5DB: List[Tuple[str, List[str], List[str], str]] = [
    ("verification_noise/2035_same_00_snr5.wav",
     ["verification-noise-5db", "noisy-5db", "librispeech", "real-human-speech", "verification-positive"],
     ["librispeech-2035", "musan-noise"],
     "speaker 2035 + MUSAN @ +5 dB"),
    ("verification_noise/1673_same_00_snr5.wav",
     ["verification-noise-5db", "noisy-5db", "librispeech", "real-human-speech", "verification-positive"],
     ["librispeech-1673", "musan-noise"],
     "speaker 1673 + MUSAN @ +5 dB"),
    ("verification_noise/84_same_00_snr5.wav",
     ["verification-noise-5db", "noisy-5db", "librispeech", "real-human-speech", "verification-positive"],
     ["librispeech-84", "musan-noise"],
     "speaker 84 + MUSAN @ +5 dB"),
    ("verification_noise/1673_diff_00_from_1988_snr5.wav",
     ["verification-noise-5db", "noisy-5db", "librispeech", "real-human-speech", "verification-negative"],
     ["librispeech-1673", "musan-noise"],
     "diff-pair 1673/1988 + MUSAN @ +5 dB"),
    ("verification_noise/84_diff_00_from_1673_snr5.wav",
     ["verification-noise-5db", "noisy-5db", "librispeech", "real-human-speech", "verification-negative"],
     ["librispeech-84", "musan-noise"],
     "diff-pair 84/1673 + MUSAN @ +5 dB"),
    ("verification_noise/2035_diff_00_from_1673_snr5.wav",
     ["verification-noise-5db", "noisy-5db", "librispeech", "real-human-speech", "verification-negative"],
     ["librispeech-2035", "musan-noise"],
     "diff-pair 2035/1673 + MUSAN @ +5 dB"),
]

VERIFICATION_NOISE_0DB: List[Tuple[str, List[str], List[str], str]] = [
    ("verification_noise/2035_same_00_snr0.wav",
     ["verification-noise-0db", "noisy-0db", "librispeech", "real-human-speech", "verification-positive"],
     ["librispeech-2035", "musan-noise", "worst-case"],
     "speaker 2035 + MUSAN @ 0 dB (worst-case noise)"),
    ("verification_noise/1673_same_00_snr0.wav",
     ["verification-noise-0db", "noisy-0db", "librispeech", "real-human-speech", "verification-positive"],
     ["librispeech-1673", "musan-noise", "worst-case"],
     "speaker 1673 + MUSAN @ 0 dB"),
    ("verification_noise/84_same_00_snr0.wav",
     ["verification-noise-0db", "noisy-0db", "librispeech", "real-human-speech", "verification-positive"],
     ["librispeech-84", "musan-noise", "worst-case"],
     "speaker 84 + MUSAN @ 0 dB"),
    ("verification_noise/1673_diff_00_from_1988_snr0.wav",
     ["verification-noise-0db", "noisy-0db", "librispeech", "real-human-speech", "verification-negative"],
     ["librispeech-1673", "musan-noise", "worst-case"],
     "diff-pair 1673/1988 + MUSAN @ 0 dB"),
    ("verification_noise/84_diff_00_from_1673_snr0.wav",
     ["verification-noise-0db", "noisy-0db", "librispeech", "real-human-speech", "verification-negative"],
     ["librispeech-84", "musan-noise", "worst-case"],
     "diff-pair 84/1673 + MUSAN @ 0 dB"),
    ("verification_noise/2035_diff_00_from_1673_snr0.wav",
     ["verification-noise-0db", "noisy-0db", "librispeech", "real-human-speech", "verification-negative"],
     ["librispeech-2035", "musan-noise", "worst-case"],
     "diff-pair 2035/1673 + MUSAN @ 0 dB"),
]

SCENARIO_MULTI_SPEAKER: List[Tuple[str, List[str], List[str], str]] = [
    ("scenario_a/sceneA_00_BAB.wav",
     ["scenario-multi-speaker", "multi-speaker", "librispeech", "real-human-speech"],
     ["librispeech", "pattern-BAB"],
     "speaker pattern B-A-B (3 turns)"),
    ("scenario_a/sceneA_01_ABABA.wav",
     ["scenario-multi-speaker", "multi-speaker", "librispeech", "real-human-speech"],
     ["librispeech", "pattern-ABABA"],
     "speaker pattern A-B-A-B-A (5 turns)"),
]

# Multilingual public-domain reads (LibriVox-derived). 2 clips × 4 langs.
# zh/ is empty in the benchmark corpus — handled separately when it lands.
MULTILINGUAL: List[Tuple[str, List[str], List[str], str]] = []
for lang in ("es", "fr", "ja", "de"):
    for idx in (0, 1):
        MULTILINGUAL.append((
            f"{lang}/pub_{lang}_{idx:02d}.wav",
            ["multilingual", f"lang-{lang}", "real-human-speech", "public-domain"],
            ["librivox-public", f"language-{lang}"],
            f"public-domain {lang.upper()} read #{idx:02d}",
        ))


ALL_CLIPS = (
    REALTIME_PROMPTS
    + ENGLISH_UTTERANCES
    + MULTILINGUAL
    + VERIFICATION_CLEAN
    + VERIFICATION_NOISE_5DB
    + VERIFICATION_NOISE_0DB
    + SCENARIO_MULTI_SPEAKER
)


# ─── Auto-taggers ─────────────────────────────────────────────────────────────
# Audio-derived tags now live in backend/ingest/auto_tag.py (Plan D Stage A1)
# so the live /ws/mic save path can call the same code. Filename-only
# heuristics (which need rel_path) stay here because they're ingest-specific.

from ingest.auto_tag import (   # noqa: E402  -- sys.path patched above
    detect_language_real as _detect_language_real_audio,
    speech_ratio,
    speech_ratio_bucket,
    count_speakers_audio,
    duration_bucket,
    estimate_snr_db,
)


def derive_language_heuristic(rel_path: str) -> str:
    """Heuristic fallback: source-dir convention. Used only when real LID
    fails or returns low confidence."""
    head = rel_path.split("/", 1)[0]
    if head in ("en", "zh", "es", "fr", "ja", "de"):
        return head
    return "en"


async def detect_language_real(
    wav_path: Path, rel_path: str, *, min_confidence: float = 0.50
) -> Tuple[str, Optional[float]]:
    """Audio LID with directory-heuristic fallback.

    Wraps the pure-audio detector from backend/ingest/auto_tag.py and
    layers on a fall-back to derive_language_heuristic(rel_path) for the
    curated corpus where the source-dir name is the labelled language.
    """
    lang, conf = await _detect_language_real_audio(
        wav_path, min_confidence=min_confidence
    )
    if lang is not None and conf is not None and conf >= min_confidence:
        return lang, conf
    # Low confidence or detector failed → fall back to source-dir heuristic.
    return derive_language_heuristic(rel_path), conf


def derive_speaker_count(rel_path: str) -> int:
    """Filename pattern → speaker count.
      sceneA_00_BAB.wav    → 2  (alphabet-pattern reveals two speakers)
      sceneA_01_ABABA.wav  → 2
      everything else      → 1
    """
    name = rel_path.rsplit("/", 1)[-1].lower()
    if "_bab" in name or "_abab" in name or "_baba" in name or "_abba" in name:
        return 2
    return 1


# ─── Ingest helpers ──────────────────────────────────────────────────────────

def purge_local(data_dir: Path) -> int:
    """Delete every clip directory under data_dir/clips/."""
    clips_root = data_dir / "clips"
    if not clips_root.exists():
        return 0
    n = 0
    for d in clips_root.iterdir():
        if d.is_dir():
            shutil.rmtree(d)
            n += 1
    return n


def post_clip(api: str, abs_path: Path, original_filename: str) -> dict:
    with open(abs_path, "rb") as f:
        files = {"file": (original_filename, f, "audio/wav")}
        data = {"source": "upload", "uploaded_by": "benchmark-ingest"}
        r = httpx.post(f"{api}/api/clips", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def patch_tags(
    api: str,
    clip_id: str,
    scenarios: List[str],
    user_tags: List[str],
    *,
    language: Optional[str] = None,
    snr_db: Optional[float] = None,
    speaker_count: Optional[int] = None,
) -> None:
    body: dict = {"scenarios": scenarios, "user_tags": user_tags}
    if language is not None:
        body["language_detected"] = language
    if snr_db is not None and snr_db == snr_db:   # not NaN
        body["snr_db"] = snr_db
    if speaker_count is not None:
        body["speaker_count_estimate"] = speaker_count
    r = httpx.patch(f"{api}/api/clips/{clip_id}", json=body, timeout=30)
    r.raise_for_status()


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", default="http://localhost:8000",
                   help="trial-app backend URL")
    p.add_argument("--purge-existing", action="store_true",
                   help="rm -rf backend/data/clips/* before ingesting")
    p.add_argument("--data-dir", default="/Users/yvette/code/audio-model-pipeline-trial/backend/data",
                   help="trial-app DATA_DIR (only used with --purge-existing)")
    p.add_argument("--no-real-lid", action="store_true",
                   help="skip the WhisperLID pass; use source-dir heuristic only "
                        "(faster but less accurate on unknown clips)")
    p.add_argument("--no-speaker-count", action="store_true",
                   help="skip Resemblyzer speaker-count clustering; use filename "
                        "heuristic (faster; degrades on arbitrary uploads)")
    args = p.parse_args()

    if not BENCHMARK_ROOT.is_dir():
        print(f"ERROR: benchmark root not found at {BENCHMARK_ROOT}", file=sys.stderr)
        return 1

    if args.purge_existing:
        n = purge_local(Path(args.data_dir))
        print(f"purged {n} existing clip directorie{'s' if n != 1 else ''}")

    # Liveness probe
    try:
        h = httpx.get(f"{args.api}/api/health", timeout=5)
        h.raise_for_status()
    except Exception as e:
        print(f"ERROR: backend not reachable at {args.api} ({e})", file=sys.stderr)
        return 2

    by_category: dict = {}
    failed = 0
    n_lid_fallback = 0
    for rel_path, scenarios, user_tags, note in ALL_CLIPS:
        src = BENCHMARK_ROOT / rel_path
        if not src.exists():
            print(f"  ✗ MISSING source: {rel_path}", file=sys.stderr)
            failed += 1
            continue
        try:
            clip = post_clip(args.api, src, src.name)
            duration_s = clip.get("duration_s")

            # ── Audio-derived auto-tagger ───────────────────────────────
            try:
                snr = estimate_snr_db(src)
            except Exception:
                snr = None
            sr = speech_ratio(src)
            sb = speech_ratio_bucket(sr)
            db = duration_bucket(duration_s)

            # Speaker count: prefer audio-derived (Resemblyzer); fall back
            # to filename heuristic if the encoder isn't available.
            spk_audio = count_speakers_audio(src) if not args.no_speaker_count else None
            speaker_count = spk_audio if spk_audio is not None else derive_speaker_count(rel_path)

            # Real LID via WhisperLIDAdapter (~200 ms on CPU per clip)
            if args.no_real_lid:
                language = derive_language_heuristic(rel_path)
                lid_conf = None
            else:
                language, lid_conf = await detect_language_real(src, rel_path)
                if lid_conf is None or lid_conf < 0.50:
                    n_lid_fallback += 1

            # ── Add audio-derived scenarios ──────────────────────────────
            scenarios = list(scenarios)
            scenarios.append(f"lang-{language}")
            if snr is not None and snr == snr:
                if   snr >= 15: scenarios.append("snr-clean")
                elif snr >=  5: scenarios.append("snr-mid")
                else:           scenarios.append("snr-noisy")
            if sb: scenarios.append(sb)
            if db: scenarios.append(db)
            if speaker_count and speaker_count >= 2:
                scenarios.append("multi-speaker")
            elif speaker_count == 1:
                scenarios.append("single-speaker")

            patch_tags(
                args.api, clip["id"], scenarios, user_tags,
                language=language, snr_db=snr, speaker_count=speaker_count,
            )
            cat = scenarios[0]
            by_category.setdefault(cat, []).append((clip["id"], src.name, note))
            conf_str = f" ({lid_conf:.2f})" if lid_conf is not None else ""
            spk_label = f"{speaker_count}{'(audio)' if spk_audio is not None else '(name)'}"
            print(f"  ✓ [{cat}] {src.name:<35}  →  {clip['id'][:8]}…  "
                  f"lang={language}{conf_str}  snr={snr if snr else '—':>5}dB  "
                  f"spk={spk_label}  speech={sr if sr is not None else '—':<5}  "
                  f"dur={db}")
        except Exception as e:
            print(f"  ✗ FAILED: {rel_path}: {e}", file=sys.stderr)
            failed += 1

    print()
    print("=" * 60)
    for cat in sorted(by_category):
        clips = by_category[cat]
        print(f"  {cat:30}  {len(clips):>3} clip{'s' if len(clips) != 1 else ''}")
    total = sum(len(v) for v in by_category.values())
    print("-" * 60)
    print(f"  {'TOTAL':30}  {total:>3} clips, {failed} failed")
    if not args.no_real_lid:
        print(f"  {'LID confidence < 0.50 → fallback':30}  "
              f"{n_lid_fallback:>3} clip{'s' if n_lid_fallback != 1 else ''}")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
