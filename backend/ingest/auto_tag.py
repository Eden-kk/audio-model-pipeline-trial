"""Audio-derived auto-tagger — shared between scripts/ and the live
mic-capture save path.

Everything here takes a WAV file (or numpy samples) and returns
acoustic features + scenario tags. No filename heuristics, no
HTTP — pure audio in, dict out, so it can run inside the FastAPI
WebSocket handler when a live-mic capture finishes.

Originally lived inside ``scripts/ingest_benchmark_corpus.py``;
factored out for Plan D Stage A1 so the ``/ws/mic`` save flow can
call ``apply_audio_tags()`` without shelling out to the script.

Functions:
    detect_language_real(wav_path)         -> (lang_code, confidence)
    speech_ratio(wav_path)                 -> 0.0..1.0 or None
    speech_ratio_bucket(ratio)             -> "mostly-speech"/"partial-speech"/"mostly-silence"
    count_speakers_audio(wav_path)         -> 1..5 or None
    duration_bucket(duration_s)            -> "short"/"medium"/"long-form"
    estimate_snr_db(wav_path)              -> float (dB)
    apply_audio_tags(wav_path, duration_s) -> dict — bundle of all of above
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ─── Language ID via faster-whisper-tiny (real, not filename-based) ─────────

_LID_ADAPTER: Any = None


def _get_lid_adapter() -> Any:
    """Lazy-load the WhisperLIDAdapter singleton.

    Imported from `backend.adapters.whisper_lid_adapter`; assumes the
    caller's sys.path already includes the backend dir (which the FastAPI
    server does by construction; the CLI script extends sys.path itself).
    """
    global _LID_ADAPTER
    if _LID_ADAPTER is None:
        try:
            from adapters.whisper_lid_adapter import WhisperLIDAdapter   # type: ignore[import]
        except ImportError:
            from backend.adapters.whisper_lid_adapter import WhisperLIDAdapter   # type: ignore[import]
        _LID_ADAPTER = WhisperLIDAdapter()
    return _LID_ADAPTER


async def detect_language_real(
    wav_path: Path | str,
    *,
    min_confidence: float = 0.50,
) -> Tuple[Optional[str], Optional[float]]:
    """Run faster-whisper-tiny LID on the audio.

    Returns (lang_code, confidence) — or (None, None) if the model fails.
    Caller decides what to do with low-confidence results; we don't fall
    back to a filename heuristic here because callers like /ws/mic don't
    have one.
    """
    try:
        adapter = _get_lid_adapter()
        out = await adapter.lid(str(wav_path), {})
        lang = out.get("language")
        conf = out.get("confidence")
        if lang is None or conf is None:
            return None, None
        return str(lang), float(conf)
    except Exception as e:
        print(f"    [auto_tag.lid] failed ({type(e).__name__}: {e})",
              file=sys.stderr)
        return None, None


# ─── Energy-VAD speech ratio ────────────────────────────────────────────────

def speech_ratio(wav_path: Path | str) -> Optional[float]:
    """Fraction of the clip that's voice activity (vs silence/noise).

    Energy VAD: 30 ms frames, classify as speech if RMS >= 2× the
    10th-percentile noise floor. Crude but robust on curated clips.
    Returns None on read errors.
    """
    import wave
    import numpy as np

    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            ch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
    except Exception:
        return None
    if sw != 2:
        return None
    samples = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)
    samples = samples.astype(np.float32) / 32768.0

    frame_len = max(1, int(sr * 0.030))   # 30 ms
    n_full = (samples.size // frame_len) * frame_len
    if n_full == 0:
        return None
    frames = samples[:n_full].reshape(-1, frame_len)
    rms = np.sqrt(np.maximum((frames ** 2).mean(axis=1), 1e-12))

    noise_floor = float(np.percentile(rms, 10))
    threshold = max(noise_floor * 2.0, 1e-4)
    speech_frames = int((rms > threshold).sum())
    return float(round(speech_frames / len(rms), 3))


def speech_ratio_bucket(ratio: Optional[float]) -> Optional[str]:
    if ratio is None:
        return None
    if ratio >= 0.80:
        return "mostly-speech"
    if ratio >= 0.40:
        return "partial-speech"
    return "mostly-silence"


# ─── Resemblyzer-based speaker count (audio-derived) ────────────────────────

_RES_ENCODER: Any = None


def count_speakers_audio(
    wav_path: Path | str, *, threshold: float = 0.70
) -> Optional[int]:
    """Distinct-speaker count via Resemblyzer windowed embeddings + greedy
    cosine clustering. Window size 1.6 s, hop 0.4 s (Resemblyzer partials).

    Returns 1 / 2 / 3+ (capped at 5) for clean clips, None on failure.
    """
    global _RES_ENCODER
    try:
        import numpy as np
        from resemblyzer import VoiceEncoder, preprocess_wav   # type: ignore[import]
    except Exception:
        return None
    try:
        if _RES_ENCODER is None:
            _RES_ENCODER = VoiceEncoder(verbose=False)

        wav = preprocess_wav(str(wav_path))
        # < ~2 s of audio can't be split into reliable partials
        if len(wav) < 16000 * 2:
            return 1

        _, partials, _ = _RES_ENCODER.embed_utterance(
            wav, return_partials=True, rate=2
        )
        if len(partials) < 2:
            return 1

        # Greedy single-pass clustering: each partial merges into the
        # nearest existing centroid if cosine >= threshold, else spawns
        # a new cluster.
        centroids = [partials[0]]
        for p in partials[1:]:
            sims = []
            for c in centroids:
                denom = float(np.linalg.norm(p) * np.linalg.norm(c))
                sims.append(float(np.dot(p, c) / denom) if denom > 0 else 0.0)
            best = max(sims)
            if best >= threshold:
                idx = sims.index(best)
                centroids[idx] = (centroids[idx] + p) / 2.0
            else:
                centroids.append(p)
        return min(len(centroids), 5)
    except Exception:
        return None


# ─── Duration & SNR (cheap, no model) ───────────────────────────────────────

def duration_bucket(duration_s: Optional[float]) -> Optional[str]:
    if duration_s is None:
        return None
    if duration_s < 5.0:
        return "short"
    if duration_s < 15.0:
        return "medium"
    return "long-form"


def estimate_snr_db(wav_path: Path | str) -> float:
    """Energy-based SNR estimate.

    50 ms frames; RMS per frame; SNR = 20·log10(p90 / p10). The
    90th-percentile-loud frame is treated as speech, the 10th-percentile
    as noise. Coarse but robust for clean (>15 dB) vs noisy (<10 dB)
    bucketing.
    """
    import wave
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        n  = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)

    if sw != 2:
        return float("nan")
    samples = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)
    samples = samples.astype(np.float32) / 32768.0

    frame_len = max(1, int(sr * 0.050))
    n_full = (samples.size // frame_len) * frame_len
    if n_full == 0:
        return float("nan")
    frames = samples[:n_full].reshape(-1, frame_len)
    rms = np.sqrt(np.maximum((frames ** 2).mean(axis=1), 1e-12))

    p90 = float(np.percentile(rms, 90))
    p10 = float(np.percentile(rms, 10))
    if p10 < 1e-9:
        return 60.0
    snr = 20.0 * (np.log10(p90) - np.log10(p10))
    return float(round(snr, 1))


def snr_bucket(snr_db: float) -> Optional[str]:
    """SNR → scenario tag. NaN-safe."""
    if snr_db != snr_db:   # NaN
        return None
    if snr_db >= 22.0:
        return "snr-clean"
    if snr_db >= 12.0:
        return "snr-mid"
    return "snr-noisy"


# ─── Bundle: one call returns all audio-derived tags ────────────────────────

async def apply_audio_tags(
    wav_path: Path | str,
    *,
    duration_s: Optional[float] = None,
    run_lid: bool = True,
    run_speaker_count: bool = True,
) -> Dict[str, Any]:
    """Single entry point used by both the ingest CLI and the /ws/mic save
    handler.

    Returns:
        {
          "language": "en"|None,           # WhisperLID top-1, may be None
          "language_confidence": 0.92|None,
          "snr_db": 18.4|None,
          "speech_ratio": 0.74|None,
          "speaker_count": 1|None,
          "duration_bucket": "short"|"medium"|"long-form"|None,
          "scenarios": [...],              # subset of:
              ["lang-{xx}", "snr-clean|snr-mid|snr-noisy",
               "mostly-speech|partial-speech|mostly-silence",
               "single-speaker|multi-speaker",
               "short|medium|long-form"]
        }

    All keys may be None if the underlying detector fails — callers
    should treat them as best-effort and fall back to whatever metadata
    they already have (filename, declared language, etc.).
    """
    scenarios: list[str] = []

    # SNR — synchronous, very cheap
    try:
        snr_db: Optional[float] = estimate_snr_db(wav_path)
        if snr_db != snr_db:   # NaN
            snr_db = None
    except Exception:
        snr_db = None
    s_bucket = snr_bucket(snr_db) if snr_db is not None else None
    if s_bucket:
        scenarios.append(s_bucket)

    # Speech ratio — synchronous, cheap
    sr = speech_ratio(wav_path)
    sr_b = speech_ratio_bucket(sr)
    if sr_b:
        scenarios.append(sr_b)

    # Speaker count — Resemblyzer, slow first call (model load) then ~200 ms
    spk: Optional[int] = None
    if run_speaker_count:
        spk = count_speakers_audio(wav_path)
        if spk is not None:
            scenarios.append("multi-speaker" if spk >= 2 else "single-speaker")

    # Duration bucket — straight metadata
    db = duration_bucket(duration_s)
    if db:
        scenarios.append(db)

    # Language — async (faster-whisper); slowest of all (~1.5 s first call)
    lang: Optional[str] = None
    lang_conf: Optional[float] = None
    if run_lid:
        lang, lang_conf = await detect_language_real(wav_path)
        if lang and lang_conf is not None and lang_conf >= 0.50:
            scenarios.append(f"lang-{lang}")

    return {
        "language": lang,
        "language_confidence": lang_conf,
        "snr_db": snr_db,
        "speech_ratio": sr,
        "speaker_count": spk,
        "duration_bucket": db,
        "scenarios": scenarios,
    }
