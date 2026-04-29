"""Native FastAPI server for MiniCPM-o-4.5 on a NVIDIA CUDA box.

Companion to `modal_app_minicpm.py`: same wire shape (`/health`,
`POST /v1/omni`, `POST /v1/omni-stream` with NDJSON streaming) so the
trial-app's MiniCPMOAdapter can talk to either backend with only a URL flip
(`MINICPM_O_REALTIME_URL`).

Why a separate file from `server.py` (NeMo): NeMo + MiniCPM-o have wildly
different deps (transformers 4.51 + minicpmo-utils + Qwen3 vs nemo_toolkit
+ megatron). They can't share a venv, so they don't share a server module
either — each lives in its own venv and gets its own uvicorn process.

Run:
    MODEL_CACHE_DIR=/raid/<you>/audio-trial-models \\
    CUDA_VISIBLE_DEVICES=7 \\
    bash model-server/run-minicpmo.sh
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ─── torchaudio.save BytesIO compat shim ────────────────────────────────────
# torchaudio 2.11 routed file I/O through TorchCodec, whose AudioEncoder
# refuses BytesIO sinks ("Couldn't allocate AVFormatContext ... Invalid
# argument"). MiniCPM-o-4.5's TTS path (`stepaudio2/token2wav.py:141`)
# calls `torchaudio.save(BytesIO, wav, sample_rate=24000, format='wav')`,
# which silently breaks audio output. Monkey-patch torchaudio.save to
# detour BytesIO destinations through soundfile.write (which natively
# supports memory buffers); leave file-path saves on the upstream code path.
import torchaudio  # noqa: E402
import io as _io  # noqa: E402

_orig_torchaudio_save = torchaudio.save


def _torchaudio_save_bytesio_compat(uri, src, sample_rate, *args, **kwargs):
    if isinstance(uri, _io.BytesIO) or isinstance(uri, _io.IOBase):
        import soundfile as _sf
        import numpy as _np
        wav = src.detach().cpu().numpy() if isinstance(src, torch.Tensor) else _np.asarray(src)
        # torchaudio is (channels, samples); soundfile is (samples, channels).
        if wav.ndim == 2:
            wav = wav.T
        fmt = (kwargs.get("format") or "WAV").upper()
        subtype = "PCM_16" if fmt in ("WAV", "WAVE") else None
        _sf.write(uri, wav, int(sample_rate), format=fmt, subtype=subtype or "PCM_16")
        return
    return _orig_torchaudio_save(uri, src, sample_rate, *args, **kwargs)


torchaudio.save = _torchaudio_save_bytesio_compat


# ─── Sentinel for cross-thread generator drain ─────────────────────────────
# StopIteration can't cross thread boundaries cleanly when the streaming
# generator runs on the executor pool; sentinel-based termination is
# robust to that.
_SENTINEL = object()


# ─── Request / response schemas ────────────────────────────────────────────
# Mirror modal_app_minicpm.py exactly — the adapter's HTTP client expects
# this shape on both lanes.

_DEFAULT_USER_INSTRUCTION = (
    "Respond to what was said. Be concise and conversational. "
    "Do NOT transcribe or repeat what was said — reply to it."
)


class OmniRequest(BaseModel):
    audio_b64: Optional[str] = Field(
        default=None,
        description="Base64 16 kHz mono PCM WAV bytes.",
    )
    image_b64: Optional[str] = Field(
        default=None,
        description="Base64 JPEG bytes. Single most-recent frame for v1.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system message prepended to the chat history.",
    )
    user_instruction: Optional[str] = Field(
        default=None,
        description=(
            "Text appended to the user turn alongside the audio / image, "
            "telling the model what to do with the media. "
            "Defaults to a short 'respond to what was said' prompt that "
            "prevents MiniCPM-o from falling back to pure ASR mode. "
            "Set to '' to suppress the instruction entirely."
        ),
    )
    history: Optional[list] = Field(
        default=None,
        description="Prior turns as MiniCPM msgs[] (list of {role, content[...]}).",
    )
    generate_audio: bool = Field(
        default=False,
        description="When true, response includes a synthesised audio_b64 reply.",
    )
    max_new_tokens: int = Field(default=256, ge=16, le=1024)
    temperature: float = Field(default=0.3, ge=0.0, le=1.5)


class OmniResponse(BaseModel):
    text: str
    audio_b64: Optional[str] = None
    sample_rate: Optional[int] = None
    latency_ms: float
    model: str = "openbmb/MiniCPM-o-4_5"


# ─── Model load (module scope, runs once at uvicorn import) ─────────────────

_MODEL_ID = os.environ.get("MINICPMO_MODEL", "openbmb/MiniCPM-o-4_5")
print(f"[setup] Loading {_MODEL_ID} …", flush=True)
_t0 = time.perf_counter()

# Late import — `transformers` is heavy and we want the print() above to
# show first so the launcher log makes the boot order obvious.
from transformers import AutoModel  # noqa: E402

# init_vision/audio/tts ALL true so the same process can serve any of the
# realtime-omni request shapes without a reload. Each init flag adds GPU
# memory + load time but keeps the API surface uniform.
model = AutoModel.from_pretrained(
    _MODEL_ID,
    trust_remote_code=True,
    attn_implementation="sdpa",   # flash_attention_2 wheels for sm_100 are patchy
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=True,
    init_tts=True,
)
model.eval().cuda()
# init_tts() is the documented separate step required for audio generation.
# Skipping it makes generate_audio=True silently produce empty output.
model.init_tts()

print(f"[setup] Ready in {time.perf_counter() - _t0:.1f}s", flush=True)


# ─── FastAPI surface ────────────────────────────────────────────────────────

app = FastAPI(title="MiniCPM-o realtime omni (native CUDA)")


def _build_msgs(req: OmniRequest) -> List[Dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    from PIL import Image

    user_content: List[Any] = []
    if req.image_b64:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
            user_content.append(img)
        except Exception as e:
            raise HTTPException(400, f"image_b64 decode failed: {e}")
    if req.audio_b64:
        try:
            raw = base64.b64decode(req.audio_b64)
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                # Cheap linear resample — quality loss negligible for ASR.
                ratio = 16000.0 / sr
                n_new = int(audio.size * ratio)
                idx = np.minimum(
                    (np.arange(n_new) / ratio).astype("int64"),
                    audio.size - 1,
                )
                audio = audio[idx].astype("float32")
            user_content.append(audio)
        except Exception as e:
            raise HTTPException(400, f"audio_b64 decode failed: {e}")

    # Append a text instruction alongside the media so MiniCPM-o generates
    # a *reply* rather than falling back to its default ASR (transcribe) mode.
    # req.user_instruction overrides the default; set it to "" to suppress.
    # Pure-text warmup turns (no audio/image) get a minimal placeholder.
    if user_content:  # media present — inject the instruction
        instruction = req.user_instruction
        if instruction is None:
            instruction = _DEFAULT_USER_INSTRUCTION
        if instruction:
            user_content.append(instruction)
    else:
        # Pure-text turn (rare; mostly for warmup / smoke-test).
        user_content.append(req.user_instruction or "Hello.")

    msgs: List[Dict[str, Any]] = []
    if req.system_prompt:
        msgs.append({"role": "system", "content": [req.system_prompt]})
    if req.history:
        msgs.extend(req.history)
    msgs.append({"role": "user", "content": user_content})
    return msgs


def _wav_to_b64(audio_arr, sr: int) -> str:
    """Encode a float32 numpy waveform → b64-encoded PCM-16 WAV."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_arr, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model": _MODEL_ID, "lane": "local-cuda"}


@app.post("/v1/omni", response_model=OmniResponse)
async def omni(req: OmniRequest) -> OmniResponse:
    """One synchronous utterance → response."""
    import soundfile as sf
    import tempfile

    t0 = time.perf_counter()
    msgs = _build_msgs(req)

    output_audio_path: Optional[str] = None
    if req.generate_audio:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        output_audio_path = tmp.name

    try:
        loop = asyncio.get_running_loop()

        def _run_chat():
            with torch.no_grad():
                return model.chat(
                    msgs=msgs,
                    do_sample=req.temperature > 0,
                    temperature=req.temperature,
                    max_new_tokens=req.max_new_tokens,
                    use_tts_template=req.generate_audio,
                    generate_audio=req.generate_audio,
                    output_audio_path=output_audio_path,
                )

        res = await loop.run_in_executor(None, _run_chat)
    except Exception as e:
        raise HTTPException(500, f"model.chat failed: {type(e).__name__}: {e}")

    text = res.get("text") if isinstance(res, dict) else str(res)

    audio_b64_out: Optional[str] = None
    sample_rate: Optional[int] = None
    if output_audio_path and os.path.exists(output_audio_path):
        try:
            audio_arr, sr = sf.read(output_audio_path, dtype="float32")
            buf = io.BytesIO()
            sf.write(buf, audio_arr, sr, format="WAV", subtype="PCM_16")
            audio_b64_out = base64.b64encode(buf.getvalue()).decode("ascii")
            sample_rate = int(sr)
        except Exception as e:
            print(f"[omni] WARNING: failed to load output audio: {e}", flush=True)
        finally:
            try:
                os.unlink(output_audio_path)
            except OSError:
                pass

    return OmniResponse(
        text=text or "",
        audio_b64=audio_b64_out,
        sample_rate=sample_rate,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        model=_MODEL_ID,
    )


@app.post("/v1/omni-stream")
async def omni_stream(req: OmniRequest):
    """Streaming variant — emits NDJSON events as the model generates.

    Two execution paths, transparent to the client:
      1. Native streaming via `model.streaming_chat(...)` if the loaded
         snapshot exposes it.
      2. Chunked-WAV fallback — runs blocking model.chat(), then chunks
         the resulting WAV into ~500 ms slices.

    Both paths emit:
      {"type":"transcript","text":"...","is_final":true}
      {"type":"text_delta","text":"..."}
      {"type":"audio_b64","data":"...","sample_rate":int}
      {"type":"done","latency_ms":float,"streaming_path":"native"|"chunked"}
    """
    import json as _json
    import numpy as np
    import soundfile as sf
    import tempfile

    t0 = time.perf_counter()
    msgs = _build_msgs(req)

    async def event_stream():
        streaming_path = "native"
        streaming_chat = getattr(model, "streaming_chat", None)

        # ── Path 1: native streaming_chat ────────────────────────────────
        # streaming_chat works for both text-only and audio generation;
        # use it whenever the method exists (regardless of generate_audio).
        if callable(streaming_chat):
            try:
                full_text = ""

                def _run_streaming():
                    with torch.no_grad():
                        yield from streaming_chat(
                            msgs=msgs,
                            do_sample=req.temperature > 0,
                            temperature=req.temperature,
                            max_new_tokens=req.max_new_tokens,
                            use_tts_template=True,
                            generate_audio=True,
                        )

                loop = asyncio.get_running_loop()
                gen = await loop.run_in_executor(None, lambda: iter(_run_streaming()))

                while True:
                    try:
                        chunk = await loop.run_in_executor(None, lambda: next(gen, _SENTINEL))
                    except StopIteration:
                        break
                    if chunk is _SENTINEL:
                        break

                    if isinstance(chunk, tuple) and len(chunk) == 3:
                        text_chunk, audio_chunk, sr = chunk
                    elif isinstance(chunk, dict):
                        text_chunk = chunk.get("text") or ""
                        audio_chunk = chunk.get("audio")
                        sr = chunk.get("sample_rate", 24000)
                    else:
                        raise RuntimeError(
                            f"streaming_chat yielded unexpected type {type(chunk).__name__}"
                        )

                    if text_chunk:
                        full_text += text_chunk
                        yield _json.dumps({"type": "text_delta", "text": text_chunk}) + "\n"
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        yield _json.dumps({
                            "type": "audio_b64",
                            "data": _wav_to_b64(np.asarray(audio_chunk, dtype="float32"), int(sr)),
                            "sample_rate": int(sr),
                        }) + "\n"

                if full_text:
                    yield _json.dumps({
                        "type": "transcript",
                        "text": full_text,
                        "is_final": True,
                    }) + "\n"
                yield _json.dumps({
                    "type": "done",
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "streaming_path": streaming_path,
                }) + "\n"
                return
            except Exception as e:
                print(f"[omni-stream] streaming_chat failed, falling back: "
                      f"{type(e).__name__}: {e}", flush=True)
                streaming_path = "chunked"

        # ── Path 2: chunked-WAV fallback ─────────────────────────────────
        streaming_path = "chunked"
        output_audio_path: Optional[str] = None
        if req.generate_audio:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            output_audio_path = tmp.name
        try:
            loop = asyncio.get_running_loop()

            def _run_chat():
                with torch.no_grad():
                    return model.chat(
                        msgs=msgs,
                        do_sample=req.temperature > 0,
                        temperature=req.temperature,
                        max_new_tokens=req.max_new_tokens,
                        use_tts_template=req.generate_audio,
                        generate_audio=req.generate_audio,
                        output_audio_path=output_audio_path,
                    )

            res = await loop.run_in_executor(None, _run_chat)
        except Exception as e:
            yield _json.dumps({
                "type": "done",
                "error": f"model.chat failed: {type(e).__name__}: {e}",
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "streaming_path": streaming_path,
            }) + "\n"
            return

        text = res.get("text") if isinstance(res, dict) else str(res)
        if text:
            yield _json.dumps({
                "type": "transcript", "text": text, "is_final": True,
            }) + "\n"
            yield _json.dumps({"type": "text_delta", "text": text}) + "\n"

        if output_audio_path and os.path.exists(output_audio_path):
            try:
                audio_arr, sr = sf.read(output_audio_path, dtype="float32")
                if audio_arr.ndim > 1:
                    audio_arr = audio_arr.mean(axis=1)
                # Chunk into ~500 ms slices.
                chunk_samples = max(1, int(0.5 * sr))
                for i in range(0, len(audio_arr), chunk_samples):
                    slice_ = audio_arr[i:i + chunk_samples]
                    yield _json.dumps({
                        "type": "audio_b64",
                        "data": _wav_to_b64(slice_, int(sr)),
                        "sample_rate": int(sr),
                    }) + "\n"
            except Exception as e:
                print(f"[omni-stream] WAV chunking failed: {e}", flush=True)
            finally:
                try:
                    os.unlink(output_audio_path)
                except OSError:
                    pass

        yield _json.dumps({
            "type": "done",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "streaming_path": streaming_path,
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
