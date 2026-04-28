"""Heuristic scenario auto-tagger.

Reads a clip's audio off disk, runs a pile of cheap (numpy-only) acoustic
features over it, and returns the subset of `SCENARIO_PALETTE` that the
audio fits. Optionally runs Whisper-LID to catch `code-switch` clips
(top-2 candidate languages both above a threshold).

Designed to be called from the Corpus page's "Auto-tag all" button so the
user doesn't have to click chips manually for every clip. Keep it cheap:
no GPU, no models bigger than tiny-Whisper, < 1 s per clip on CPU.

Tag detection rules (all numeric thresholds tuned for ~30 s ambient clips
at 16 kHz mono — happy to recalibrate as we accumulate ground truth):

    phone-call       : sample_rate ≤ 8000 Hz OR audio bandwidth ≤ 3.7 kHz
                       (telephone-grade is the easiest scenario to detect
                       reliably; codec irrevocably lops off > 3.4 kHz)

    quiet-office     : SNR > 22 dB AND median noise-floor energy is low.
                       i.e. mostly silence with the occasional clean voice.

    noisy-restaurant : SNR < 10 dB AND spectral centroid in voice range
                       (~1-3 kHz) — babble noise sits in the same band as
                       speech, distinct from outdoor-traffic which is lower.

    outdoor-traffic  : SNR < 12 dB AND spectral centroid < 800 Hz
                       (engine rumble + wind in the low end)

    whisper-voice    : RMS energy < 0.012 over voiced frames (very quiet
                       speech; we don't try to discriminate from background
                       silence — combined with the speech-presence check
                       it's still useful)

    multi-speaker    : Resemblyzer embedding centroid spread across
                       half-second windows > 0.35 cosine — i.e. the speaker
                       embedding wanders far enough that it can't be one
                       person. Requires Resemblyzer adapter loaded.

    code-switch      : Whisper-LID's top-2 candidates both have probability
                       > 0.20 (canonical: top-1 0.7 + top-2 0.05 means one
                       language; top-1 0.55 + top-2 0.30 means code-switch)

    indoor-cafe      : SNR 10-22 dB AND spectral centroid 600-1500 Hz
                       (mid-noise environment; weaker signal than restaurant,
                       wider band than outdoor-traffic)

    accented         : NOT auto-detected — too subjective without a per-language
                       phonetic model. Left for manual tagging.

    wearer-walking   : NOT auto-detected in v1 — needs IMU or footstep-
                       periodicity detection that's beyond a simple FFT.

The function returns a dict shaped like
    {
        "scenarios": ["quiet-office", "code-switch"],   # detected labels
        "features":  {"snr_db": 24.7, "centroid_hz": 1400.0, …},
        "evidence":  {"quiet-office": "SNR=24.7dB > 22 + low noise floor", …},
    }
so the UI can show "why" each tag fired (tooltips / dev-mode panel).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("trial-app.autotagger")


# Mirror of the frontend's SCENARIO_PALETTE — auto-tagger may only emit
# labels from this list, so the chip toggle states stay consistent.
SCENARIO_PALETTE = [
    "outdoor-traffic", "indoor-cafe", "multi-speaker", "code-switch",
    "accented", "whisper-voice", "phone-call", "noisy-restaurant",
    "quiet-office", "wearer-walking",
]


# ─── Acoustic feature extraction ─────────────────────────────────────────────


def _load_mono(audio_path: str, target_sr: int = 16000) -> Tuple["Any", int, int]:
    """Return (samples_float32, original_sample_rate, original_channels).

    We keep the *original* sample rate and channel count for the
    sample-rate / phone-call check — but the returned waveform is mono and
    resampled (cheap linear) to `target_sr` for everything downstream.
    """
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    orig_sr = sr
    orig_channels = 1 if audio.ndim == 1 else audio.shape[1]
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")
    if sr != target_sr and audio.size > 0:
        ratio = target_sr / sr
        n_new = int(audio.size * ratio)
        idx = np.minimum(
            (np.arange(n_new) / ratio).astype("int64"),
            audio.size - 1,
        )
        audio = audio[idx].astype("float32")
    return audio, orig_sr, orig_channels


def _voice_activity_mask(audio: "Any", sr: int, frame_ms: int = 30,
                         energy_pct: float = 0.30) -> "Any":
    """Adaptive energy-threshold VAD. Frames with RMS above the
    `energy_pct` percentile of the clip's RMS distribution are voice;
    rest are noise floor. Works fine for SNR estimation without needing
    webrtcvad as a dependency.
    """
    import numpy as np
    n = max(1, int(sr * frame_ms / 1000))
    if audio.size < n:
        return np.zeros(1, dtype=bool)
    n_frames = audio.size // n
    frames = audio[: n_frames * n].reshape(n_frames, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    if n_frames < 5:
        return rms > rms.mean()
    threshold = float(np.quantile(rms, energy_pct))
    return rms > max(threshold, 1e-4)


def _snr_db(audio: "Any", sr: int) -> float:
    """Crude segmental SNR: voiced-frame power vs unvoiced-frame power, dB.
    Returns +inf for completely-silent noise (clean clip), -inf if all
    frames cluster (no signal differentiation)."""
    import numpy as np
    n = max(1, int(sr * 0.030))
    if audio.size < n * 4:
        return 0.0
    n_frames = audio.size // n
    frames = audio[: n_frames * n].reshape(n_frames, n)
    powers = (frames ** 2).mean(axis=1) + 1e-12
    voiced = _voice_activity_mask(audio, sr)
    if voiced.size != powers.size:
        m = min(voiced.size, powers.size)
        voiced = voiced[:m]; powers = powers[:m]
    sig = float(powers[voiced].mean()) if voiced.any() else 0.0
    noise = float(powers[~voiced].mean()) if (~voiced).any() else sig / 1e6
    if sig <= 0 or noise <= 0:
        return 0.0
    return 10.0 * math.log10(sig / noise)


def _spectral_centroid_hz(audio: "Any", sr: int) -> float:
    """Mean spectral centroid (Hz) over the whole clip. Cheap proxy for
    'is this rumble vs babble vs clean speech'."""
    import numpy as np
    if audio.size < 512:
        return 0.0
    n = 1024
    hop = 512
    n_frames = max(1, (audio.size - n) // hop)
    if n_frames <= 0:
        return 0.0
    win = np.hanning(n).astype("float32")
    centroids = []
    for i in range(n_frames):
        seg = audio[i * hop: i * hop + n]
        if seg.size < n:
            break
        spec = np.abs(np.fft.rfft(seg * win))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        s = spec.sum() + 1e-12
        centroids.append(float((spec * freqs).sum() / s))
    return float(np.mean(centroids)) if centroids else 0.0


def _audio_bandwidth_hz(audio: "Any", sr: int) -> float:
    """Highest frequency containing meaningful energy. A telephone codec
    chops off > 3.4 kHz so this drops to ~3.5 kHz for phone-call clips
    even when the file is upsampled to 16 kHz before storage."""
    import numpy as np
    if audio.size < 1024:
        return 0.0
    spec = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    total = spec.sum() + 1e-12
    cum = np.cumsum(spec) / total
    # Bandwidth = freq below which 99% of energy lives.
    idx = int(np.searchsorted(cum, 0.99))
    idx = min(idx, freqs.size - 1)
    return float(freqs[idx])


def _voiced_rms(audio: "Any", sr: int) -> float:
    """Mean RMS over voiced frames only. Used by whisper-voice check —
    quiet speech has low voiced RMS even when overall RMS is also low."""
    import numpy as np
    n = max(1, int(sr * 0.030))
    n_frames = audio.size // n
    if n_frames < 1:
        return 0.0
    frames = audio[: n_frames * n].reshape(n_frames, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    voiced = _voice_activity_mask(audio, sr)
    if voiced.size != rms.size:
        m = min(voiced.size, rms.size); voiced = voiced[:m]; rms = rms[:m]
    return float(rms[voiced].mean()) if voiced.any() else float(rms.mean())


# ─── Optional adapter-backed signals ─────────────────────────────────────────


async def _detect_code_switch(audio_path: str, registry: Any) -> Optional[Tuple[bool, str]]:
    """Return (is_code_switch, evidence) using Whisper-LID's top-K
    distribution. None if the LID adapter isn't loaded."""
    try:
        adapter = registry.get("whisper_lid")
    except Exception:
        return None
    try:
        result = await adapter.lid(audio_path, {"top_k": 3})
    except Exception as e:
        log.warning(f"whisper_lid failed during autotag: {e}")
        return None
    cands = result.get("candidates") or []
    if len(cands) < 2:
        return False, "only one LID candidate returned"
    top1 = float(cands[0].get("confidence", 0.0))
    top2 = float(cands[1].get("confidence", 0.0))
    is_cs = top2 > 0.20 and top1 < 0.85
    ev = (
        f"top1={cands[0].get('language')}({top1:.2f}) "
        f"top2={cands[1].get('language')}({top2:.2f})"
    )
    return is_cs, ev


async def _detect_multi_speaker(audio_path: str, registry: Any) -> Optional[Tuple[bool, str]]:
    """Use Resemblyzer to extract one embedding per ~1.5s window, measure
    the spread (mean cosine distance from the centroid). High spread →
    multiple speakers."""
    try:
        adapter = registry.get("resemblyzer")
    except Exception:
        return None
    try:
        import numpy as np
        import soundfile as sf
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
        wav = preprocess_wav(audio_path)
        # Resemblyzer's embed_utterance can return per-window embeddings
        # via embed_speaker; we use the simpler manual-window approach
        # so the dependency surface stays predictable.
        encoder = adapter._encoder if hasattr(adapter, "_encoder") else VoiceEncoder()
        # Window over 1.5 s, hop 0.75 s
        sr_target = 16000
        win_s = 1.5; hop_s = 0.75
        n_win = int(win_s * sr_target); n_hop = int(hop_s * sr_target)
        if wav.size < n_win:
            return False, "audio too short for multi-speaker check"
        embs = []
        for i in range(0, wav.size - n_win + 1, n_hop):
            try:
                e = encoder.embed_utterance(wav[i: i + n_win])
                embs.append(e)
            except Exception:
                continue
        if len(embs) < 4:
            return False, f"only {len(embs)} valid windows"
        E = np.stack(embs)
        centroid = E.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        Enorm = E / norms
        cos_dist = 1.0 - (Enorm @ centroid)
        spread = float(cos_dist.mean())
        is_multi = spread > 0.35
        return is_multi, f"embedding spread={spread:.3f} (>0.35 → multi)"
    except Exception as e:
        log.warning(f"resemblyzer multi-speaker check failed: {e}")
        return None


# ─── Top-level entry point ───────────────────────────────────────────────────


async def autotag_clip(
    audio_path: str,
    *,
    registry: Any = None,
    use_lid: bool = True,
    use_speaker_spread: bool = True,
) -> Dict[str, Any]:
    """Score one clip; return detected scenarios + features + evidence.

    `registry` is the adapter registry from main.py — needed only for the
    LID + multi-speaker checks. If None or those adapters aren't loaded,
    those tags simply won't be detected (clean degradation).
    """
    audio, orig_sr, orig_channels = _load_mono(audio_path)
    if audio.size == 0:
        return {"scenarios": [], "features": {}, "evidence": {"_": "empty audio"}}

    sr = 16000  # _load_mono resamples
    snr = _snr_db(audio, sr)
    centroid = _spectral_centroid_hz(audio, sr)
    bandwidth = _audio_bandwidth_hz(audio, sr)
    vrms = _voiced_rms(audio, sr)

    features: Dict[str, Any] = {
        "orig_sample_rate": orig_sr,
        "orig_channels": orig_channels,
        "snr_db": round(snr, 2),
        "centroid_hz": round(centroid, 1),
        "bandwidth_hz": round(bandwidth, 1),
        "voiced_rms": round(vrms, 4),
        "duration_s": round(audio.size / sr, 2),
    }
    scenarios: List[str] = []
    evidence: Dict[str, str] = {}

    # phone-call
    if orig_sr <= 8000 or bandwidth < 3700:
        scenarios.append("phone-call")
        evidence["phone-call"] = (
            f"orig_sr={orig_sr}Hz, bandwidth={bandwidth:.0f}Hz "
            f"(≤8000 or <3700 → phone codec)"
        )

    # quiet-office  (high SNR, calm)
    if snr > 22 and vrms < 0.10:
        scenarios.append("quiet-office")
        evidence["quiet-office"] = f"SNR={snr:.1f}dB > 22 and voiced_rms={vrms:.3f} (calm)"

    # outdoor-traffic vs noisy-restaurant vs indoor-cafe
    if snr < 12 and centroid < 800:
        scenarios.append("outdoor-traffic")
        evidence["outdoor-traffic"] = (
            f"SNR={snr:.1f}dB < 12 and centroid={centroid:.0f}Hz < 800 (low rumble)"
        )
    elif snr < 10 and 1000 <= centroid <= 3000:
        scenarios.append("noisy-restaurant")
        evidence["noisy-restaurant"] = (
            f"SNR={snr:.1f}dB < 10 and centroid={centroid:.0f}Hz in babble band"
        )
    elif 10 <= snr <= 22 and 600 <= centroid <= 1500:
        scenarios.append("indoor-cafe")
        evidence["indoor-cafe"] = (
            f"SNR={snr:.1f}dB in 10-22 and centroid={centroid:.0f}Hz mid-band"
        )

    # whisper-voice
    if vrms < 0.012 and snr > 8:   # quiet speech, not just silence
        scenarios.append("whisper-voice")
        evidence["whisper-voice"] = f"voiced_rms={vrms:.4f} < 0.012 and SNR={snr:.1f}dB > 8"

    # code-switch (LID-backed, optional)
    if use_lid and registry is not None:
        cs = await _detect_code_switch(audio_path, registry)
        if cs:
            is_cs, ev = cs
            if is_cs:
                scenarios.append("code-switch")
            evidence["code-switch"] = ev + (" → code-switch" if is_cs else " → mono")

    # multi-speaker (Resemblyzer-backed, optional)
    if use_speaker_spread and registry is not None:
        ms = await _detect_multi_speaker(audio_path, registry)
        if ms:
            is_ms, ev = ms
            if is_ms:
                scenarios.append("multi-speaker")
            evidence["multi-speaker"] = ev + (" → multi" if is_ms else " → single")

    # Dedupe in case any rule fired twice (defensive).
    scenarios = list(dict.fromkeys(scenarios))
    return {"scenarios": scenarios, "features": features, "evidence": evidence}
