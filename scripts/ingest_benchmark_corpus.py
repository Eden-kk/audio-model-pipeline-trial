"""Curated benchmark-clip ingest for the trial-app corpus.

Pulls a small representative set of audio from
  ambient-deploy/benchmarks/test_clips/...
into the local trial-app via POST /api/clips, then PATCHes scenario +
user-tag chips so the Library page can filter by category.

Categories shipped (35 clips total — keep small enough to iterate fast):

  realtime-prompts        10  — q01..q10 (Kokoro-TTS assistant queries)
  english-utterances       5  — test_01..test_05 (longer English samples)
  verification-clean       6  — LibriSpeech same/diff pairs (clean)
  verification-noise-5db   6  — same pairs + MUSAN @ +5 dB SNR
  verification-noise-0db   6  — same pairs + MUSAN @  0 dB SNR (worst-case)
  scenario-multi-speaker   2  — interleaved BAB / ABABA patterns

Each clip gets:
  scenarios=[primary_category, ...secondary_tags]
  user_tags=[provenance/source-info]

Usage:
  python3 scripts/ingest_benchmark_corpus.py [--api http://localhost:8000]
                                             [--purge-existing]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import httpx


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


ALL_CLIPS = (
    REALTIME_PROMPTS
    + ENGLISH_UTTERANCES
    + VERIFICATION_CLEAN
    + VERIFICATION_NOISE_5DB
    + VERIFICATION_NOISE_0DB
    + SCENARIO_MULTI_SPEAKER
)


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


def patch_tags(api: str, clip_id: str, scenarios: List[str], user_tags: List[str]) -> None:
    r = httpx.patch(
        f"{api}/api/clips/{clip_id}",
        json={"scenarios": scenarios, "user_tags": user_tags},
        timeout=30,
    )
    r.raise_for_status()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", default="http://localhost:8000",
                   help="trial-app backend URL")
    p.add_argument("--purge-existing", action="store_true",
                   help="rm -rf backend/data/clips/* before ingesting")
    p.add_argument("--data-dir", default="/Users/yvette/code/audio-model-pipeline-trial/backend/data",
                   help="trial-app DATA_DIR (only used with --purge-existing)")
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
    for rel_path, scenarios, user_tags, note in ALL_CLIPS:
        src = BENCHMARK_ROOT / rel_path
        if not src.exists():
            print(f"  ✗ MISSING source: {rel_path}", file=sys.stderr)
            failed += 1
            continue
        try:
            clip = post_clip(args.api, src, src.name)
            patch_tags(args.api, clip["id"], scenarios, user_tags)
            cat = scenarios[0]
            by_category.setdefault(cat, []).append((clip["id"], src.name, note))
            print(f"  ✓ [{cat}] {src.name}  →  {clip['id']}")
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
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
