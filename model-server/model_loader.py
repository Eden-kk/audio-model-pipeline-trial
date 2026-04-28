"""Lazy model loader for the trial-app model-server.

Each model has a load() function that returns a callable
`transcribe(audio_path) -> dict`.  Models are loaded on first request and
cached forever in this process.

Loader status:
  - parakeet-tdt-1.1b: ASRModel + standard transcribe path.
  - canary-1b-flash:   EncDecMultiTaskModel + multitask transcribe(source_lang,
                       target_lang, task='asr', pnc='yes').
  - canary-qwen-2.5b:  speechlm2.SALM + generate(...) — see _load_canary_qwen.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any, Callable, Dict, Optional, Tuple


# ─── PyTorch / NeMo compatibility shims ─────────────────────────────────────
# NeMo 2.4 imports `from torch.distributed.fsdp import fully_shard`, but in
# torch 2.5.x that symbol lives under `_composable.fsdp.fully_shard`.  The
# move only happened in torch 2.6+.  Re-export at module-load time so any
# downstream `from torch.distributed.fsdp import fully_shard` sees it.
try:
    import torch.distributed.fsdp as _fsdp  # type: ignore
    if not hasattr(_fsdp, "fully_shard"):
        try:
            from torch.distributed._composable.fsdp import (
                fully_shard as _fully_shard,  # type: ignore
            )
            _fsdp.fully_shard = _fully_shard  # type: ignore[attr-defined]
        except Exception:
            pass
except Exception:
    pass

_LOADED: Dict[str, Tuple[Any, str]] = {}  # model_id → (instance, version_string)
_LOAD_LOCK = threading.Lock()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_nemo_output(out: Any) -> Tuple[str, list]:
    """Pull text + word-timing out of NeMo's transcribe() return shape.

    Newer NeMo versions return a *tuple* of (best_hypotheses, n_best); each
    element is a list, and each list-element is either a Hypothesis object
    (with .text and possibly .words) or a plain string.  We progressively
    unwrap until we land on something with .text or a string.
    """
    # Unwrap tuple (best_hypotheses, n_best)
    if isinstance(out, tuple):
        out = out[0] if out else None
    # Unwrap list-of-list (multi-clip x N-best, but we only sent one clip)
    while isinstance(out, list) and out:
        out = out[0]
    if out is None:
        return "", []

    text = ""
    words: list = []
    if isinstance(out, str):
        text = out
    elif hasattr(out, "text"):
        text = out.text or ""
        if hasattr(out, "words") and out.words:
            for w in out.words:
                words.append({
                    "word": getattr(w, "word", ""),
                    "start": float(getattr(w, "start", 0.0) or 0.0),
                    "end": float(getattr(w, "end", 0.0) or 0.0),
                })
    return text, words


# ─── Parakeet-TDT-1.1B ───────────────────────────────────────────────────────

def _load_parakeet() -> Tuple[Any, str]:
    from nemo.collections.asr.models import ASRModel  # type: ignore[import]
    model = ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
    model.eval()
    return model, "nvidia/parakeet-tdt-1.1b"


def _transcribe_parakeet(model: Any, audio_path: str) -> Dict[str, Any]:
    out = model.transcribe([audio_path], batch_size=1, verbose=False)
    text, words = _normalize_nemo_output(out)
    return {"text": text, "words": words, "language": "en"}


# ─── Canary-1B-flash (multilingual EN/DE/ES/FR) ─────────────────────────────
# Uses EncDecMultiTaskModel per HF model card — Canary's multitask config
# isn't compatible with the generic ASRModel loader.

def _load_canary_1b_flash() -> Tuple[Any, str]:
    from nemo.collections.asr.models import EncDecMultiTaskModel  # type: ignore[import]
    model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-flash")
    model.eval()
    return model, "nvidia/canary-1b-flash"


def _transcribe_canary_1b_flash(model: Any, audio_path: str) -> Dict[str, Any]:
    """Canary's transcribe() requires source_lang + target_lang + task + pnc."""
    out = model.transcribe(
        [audio_path],
        batch_size=1,
        source_lang="en",
        target_lang="en",
        task="asr",
        pnc="yes",
        verbose=False,
    )
    text, words = _normalize_nemo_output(out)
    return {"text": text, "words": words, "language": "en"}


# ─── Canary-Qwen-2.5B (EN, long-form) ────────────────────────────────────────
# Lives in nemo.collections.speechlm2 (NeMo ≥2.2 — we pin 2.4.0 in modal_app.py).

def _load_canary_qwen_25b() -> Tuple[Any, str]:
    from nemo.collections.speechlm2.models import SALM  # type: ignore[import]
    model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
    model.eval()
    return model, "nvidia/canary-qwen-2.5b"


def _transcribe_canary_qwen_25b(model: Any, audio_path: str) -> Dict[str, Any]:
    """SALM.generate signature is:
         generate(prompts, audios=Tensor, audio_lens=Tensor, ...)
       The audio data passes as a *separate* tensor argument; the prompt
       only carries the text instruction.  Inspect the deployed signature
       at GET /salm-debug if anything below drifts from upstream.
    """
    import soundfile as sf  # type: ignore
    import torch  # type: ignore

    # Load + resample (SALM expects mono float32 @ model.sampling_rate, default 16k)
    samples, sr = sf.read(audio_path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    target_sr = int(getattr(model, "sampling_rate", 16000))
    if sr != target_sr:
        try:
            import scipy.signal  # type: ignore
            samples = scipy.signal.resample_poly(samples, target_sr, sr).astype("float32")
        except Exception:
            pass

    audios = torch.from_numpy(samples).unsqueeze(0).to(model.device)   # (1, T)
    audio_lens = torch.tensor([audios.shape[1]], device=model.device)

    # Embed the audio-locator tag inside the message string. SALM's
    # replace_placeholders_and_build_targets replaces this exact token with
    # the encoded audio frames at generate-time. The Qwen prompt template
    # only has one slot ("message"); there is no separate "audio" slot.
    locator = getattr(model.cfg, "audio_locator_tag", "<|audioplaceholder|>")
    prompts = [[
        {"role": "user", "slots": {
            "message": f"{locator} Transcribe the spoken content into written text.",
        }},
    ]]
    # max_new_tokens is forwarded to self.llm.generate via **generation_kwargs.
    # 256 tokens covers ~ 30 seconds of speech worth of transcript.
    token_ids = model.generate(prompts, audios=audios, audio_lens=audio_lens,
                               max_new_tokens=256)

    # Decode token ids → text via the model's tokenizer.
    text = ""
    try:
        # token_ids shape: (batch, seq) — take the first row
        ids = token_ids[0].tolist() if token_ids.dim() > 1 else token_ids.tolist()
        text = model.tokenizer.ids_to_text(ids) if hasattr(model.tokenizer, "ids_to_text") \
               else model.tokenizer.decode(ids)
        # Strip chat markers + echoed prompt.
        marker = "assistant\n"
        if marker in text:
            text = text.split(marker, 1)[1]
        for end_marker in ("<|im_end|>", "<|endoftext|>"):
            if end_marker in text:
                text = text.split(end_marker, 1)[0]
        # SALM-Qwen sometimes echoes the user instruction at the tail of the
        # answer ("...transcript here. Transcribe the spoken content..."). Strip.
        for echo in (
            "Transcribe the spoken content into written text.",
            "Transcribe the spoken content into written text",
            "Transcribe the audio.",
            "Transcribe the following audio into written text.",
        ):
            if text.rstrip().endswith(echo):
                text = text.rstrip()[: -len(echo)]
        text = text.strip(" .,;:")
        text = text.strip()
    except Exception as e:
        text = f"[decode error: {e}]"

    return {"text": text, "words": [], "language": "en"}


# ─── Registry ─────────────────────────────────────────────────────────────────

LOADERS: Dict[str, Callable[[], Tuple[Any, str]]] = {
    "parakeet-tdt-1.1b": _load_parakeet,
    "canary-1b-flash": _load_canary_1b_flash,
    "canary-qwen-2.5b": _load_canary_qwen_25b,
}

TRANSCRIBERS: Dict[str, Callable[[Any, str], Dict[str, Any]]] = {
    "parakeet-tdt-1.1b": _transcribe_parakeet,
    "canary-1b-flash": _transcribe_canary_1b_flash,
    "canary-qwen-2.5b": _transcribe_canary_qwen_25b,
}


def status() -> Dict[str, Any]:
    out = {}
    for mid in LOADERS:
        loaded = mid in _LOADED
        out[mid] = {
            "loaded": loaded,
            "version": _LOADED[mid][1] if loaded else None,
            "transcriber_implemented": mid in TRANSCRIBERS,
        }
    return out


def get_or_load(model_id: str) -> Any:
    if model_id not in LOADERS:
        raise KeyError(f"unknown model id: {model_id!r}")
    if model_id in _LOADED:
        return _LOADED[model_id][0]
    with _LOAD_LOCK:
        if model_id in _LOADED:
            return _LOADED[model_id][0]
        instance, version = LOADERS[model_id]()
        _LOADED[model_id] = (instance, version)
        return instance


def transcribe(model_id: str, audio_path: str) -> Dict[str, Any]:
    if model_id not in TRANSCRIBERS:
        raise NotImplementedError(
            f"transcriber for {model_id!r} not implemented yet"
        )
    instance = get_or_load(model_id)
    t0 = time.perf_counter()
    out = TRANSCRIBERS[model_id](instance, audio_path)
    out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
    out["model"] = model_id
    return out
