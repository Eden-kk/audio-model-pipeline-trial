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
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

# Strip these chat-template / TTS markers when they leak into stream=True
# text deltas (MiniCPM-o-4.5 emits them as part of the final delta).
_STRIP_TOKENS = ("<|tts_eos|>", "<|im_end|>", "<|endoftext|>", "<|tts_bos|>")


def _strip_special(text: str) -> str:
    for tok in _STRIP_TOKENS:
        text = text.replace(tok, "")
    return text


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

# init_token2wav_cache() is required by streaming_generate (the new
# /ws/omni-stream path). Without it, streaming_generate raises
# `'NoneType' object is not subscriptable` at line 2329 of
# modeling_minicpmo.py because `self.token2wav_cache` is None. The
# upstream API expects a `prompt_speech_16k` reference waveform — for the
# default voice we hand it 1 s of silence, which works as a "no voice
# clone" placeholder. The chunked path doesn't need this (chat() doesn't
# touch token2wav_cache for the s3tokenizer streaming codepath).
import numpy as _np_init  # noqa: E402
try:
    model.init_token2wav_cache(_np_init.zeros(16000, dtype=_np_init.float32))
    print("[setup] init_token2wav_cache OK (default-voice silence prompt)", flush=True)
except Exception as _e:
    print(f"[setup] init_token2wav_cache failed: {type(_e).__name__}: {_e}; "
          f"/ws/omni-stream will be unavailable", flush=True)

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
                    use_tts_template=True,
                    generate_audio=req.generate_audio,
                    omni_mode=True,
                    output_audio_path=output_audio_path,
                    num_beams=1,
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
        # MiniCPM-o-4.5 doesn't expose `streaming_chat`. The primitive we
        # use is `model.chat(stream=True, ...)` — but stream=True yields
        # ONLY incremental text (the TTS audio decoder needs the full
        # finalized text to render, so it's incompatible with stream=True
        # in chat()). When generate_audio=True we must fall through to
        # the blocking Path 2.
        # use_tts_template is auto-set to True inside chat() whenever audio
        # is in the user content (modeling_minicpmo.py:1142), so we don't
        # need to force it here — but we DO need the canonical
        # audio-assistant system prompt (handed in by the adapter) to keep
        # the model in dialogue mode rather than its default ASR mode.
        streaming_path = "stream"
        try:
            if req.generate_audio:
                # Skip the streaming path entirely — fall through to the
                # blocking model.chat() path that supports TTS rendering.
                raise RuntimeError("generate_audio=True → use blocking path")
            full_text = ""
            sample_rate_hint = 24000

            def _run_streaming():
                with torch.no_grad():
                    yield from model.chat(
                        msgs=msgs,
                        do_sample=req.temperature > 0,
                        temperature=req.temperature,
                        max_new_tokens=req.max_new_tokens,
                        use_tts_template=True,
                        generate_audio=req.generate_audio,
                        omni_mode=True,
                        stream=True,
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

                text_chunk: str = ""
                audio_chunk = None
                sr = sample_rate_hint
                if isinstance(chunk, str):
                    text_chunk = chunk
                elif isinstance(chunk, tuple):
                    if len(chunk) >= 1 and isinstance(chunk[0], str):
                        text_chunk = chunk[0]
                    if len(chunk) >= 2 and chunk[1] is not None:
                        audio_chunk = chunk[1]
                    if len(chunk) >= 3 and isinstance(chunk[2], (int, float)):
                        sr = int(chunk[2])
                elif isinstance(chunk, dict):
                    text_chunk = chunk.get("text") or ""
                    audio_chunk = chunk.get("audio")
                    sr = int(chunk.get("sample_rate", sample_rate_hint))
                else:
                    # Unknown chunk shape — log once and skip rather than
                    # tearing down the whole stream.
                    print(f"[omni-stream] unexpected chunk type "
                          f"{type(chunk).__name__}; skipping", flush=True)
                    continue

                if text_chunk:
                    text_chunk = _strip_special(text_chunk)
                if text_chunk:
                    full_text += text_chunk
                    yield _json.dumps({"type": "text_delta", "text": text_chunk}) + "\n"
                    # Yield control so FastAPI/uvicorn flushes this NDJSON
                    # line to the client immediately rather than buffering
                    # until the generator is exhausted.
                    await asyncio.sleep(0)
                if audio_chunk is not None and len(audio_chunk) > 0:
                    yield _json.dumps({
                        "type": "audio_b64",
                        "data": _wav_to_b64(np.asarray(audio_chunk, dtype="float32"), int(sr)),
                        "sample_rate": int(sr),
                    }) + "\n"
                    await asyncio.sleep(0)

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
            print(f"[omni-stream] stream=True path failed, falling back: "
                  f"{type(e).__name__}: {e}", flush=True)
            streaming_path = "chunked"

        # ── Path 2: chunked-WAV fallback (blocking model.chat) ───────────
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
                        use_tts_template=True,
                        generate_audio=req.generate_audio,
                        omni_mode=True,
                        output_audio_path=output_audio_path,
                        # num_beams=1 forces sampling-only inference. The
                        # default num_beams=3 (beam search) optimises for
                        # likelihood and biases the model toward echoing
                        # the user audio in the TTS slot. The streaming
                        # path applies the same override automatically
                        # when stream=True; we mirror it here.
                        num_beams=1,
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


# Default system prompt for /ws/omni-stream when the client doesn't send one.
# Mirrors the adapter's canonical audio_assistant prompt — the magic
# `<reserved_53>` voice token + dialogue suffix are what keep the model out
# of ASR-mode (see backend/adapters/minicpm_o_adapter.py).
_DEFAULT_SYSTEM_PROMPT = (
    "Use the <reserved_53> voice. "
    "Please assist users while maintaining this voice style. "
    "Please answer the user's questions seriously and in a high quality. "
    "Please chat with the user in a highly human-like and oral style. "
    "You are a helpful assistant developed by ModelBest: MiniCPM-Omni."
)


# ─── /ws/omni-stream — true streaming via streaming_prefill / streaming_generate ──
#
# Single-tenant WebSocket endpoint that exposes the upstream MiniCPM-o
# session-based streaming API. See the realtime-fixes plan at
# ~/.claude/plans/now-for-current-repo-starry-wren.md for the full design.
#
# Wire shape per the adapter: client opens WS, sends one JSON `start`
# control frame carrying `system_prompt`, then a stream of:
#   - binary frames = raw 16 kHz mono PCM-16 audio chunks → streaming_prefill.
#   - JSON `{"event":"flush"}`            → streaming_generate, NDJSON-yield events back.
#   - JSON `{"event":"interrupt"}`        → abort generation, restore snapshot.
#   - JSON `{"event":"close"}`            → tear down session.
#
# Server emits text JSON events: `text_delta`, `audio_b64`, `done`,
# `OmniError` — same shapes the chunked path uses.

# Single global concurrency gate. Only one WS owns the model at a time.
_stream_lock = asyncio.Lock()
# Single-worker executor — keeps streaming_prefill / streaming_generate /
# reset_session / restore_speculative_snapshot strictly serialised.
_stream_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minicpmo-stream")


def _strip_special_keep_specials_for_log(s: str) -> str:
    """Same as _strip_special but only used for logs. Currently identical."""
    return _strip_special(s)


@app.websocket("/ws/omni-stream")
async def ws_omni_stream(ws: WebSocket):
    """True real-time omni dialogue. Single-tenant; second concurrent
    connection is rejected with WS close 1013 ("Try Again Later")."""
    await ws.accept()
    if _stream_lock.locked():
        try:
            await ws.send_json({
                "event": "OmniError",
                "error": "model busy — only one streaming session at a time",
            })
        except Exception:
            pass
        # 1013 = "Try Again Later" (RFC 6455 §7.4.2)
        await ws.close(code=1013)
        return

    async with _stream_lock:
        loop = asyncio.get_running_loop()
        session_id = uuid.uuid4().hex
        abort_event = asyncio.Event()
        system_prompt: Optional[str] = None
        first_audio_seen = False
        has_snapshot = False

        # streaming_prefill needs ≥ 1 s of audio per call (the model's
        # audio encoder is a Conv1d stack; sub-1s chunks underflow the
        # kernel and raise "Kernel size can't be greater than actual
        # input size"). audio_chunk_length=1.0 in the config.
        # Strategy: accumulate raw PCM in `pcm_accumulator` (bytes). When
        # we have ≥ CHUNK_SAMPLES samples, slice off CHUNK_SAMPLES and
        # send to streaming_prefill with is_last=False. On flush, send
        # whatever remains (padded if too short) with is_last=True.
        SR = 16000
        CHUNK_SAMPLES = SR * 1   # 1 s
        CHUNK_BYTES = CHUNK_SAMPLES * 2   # PCM-16 = 2 bytes/sample
        MIN_BYTES_FOR_PREFILL = 320 * 2   # 20 ms — anything smaller pads to 1 s
        pcm_accumulator = bytearray()
        any_audio_in_turn = False

        def _bytes_to_pcm_f32(blob: bytes) -> np.ndarray:
            return np.frombuffer(blob, dtype=np.int16).astype(np.float32) / 32768.0

        def _pad_to_chunk(blob: bytes) -> bytes:
            """Zero-pad a sub-1s blob up to CHUNK_BYTES so the encoder
            doesn't underflow. Used only on the last chunk at flush time."""
            if len(blob) >= CHUNK_BYTES:
                return blob
            return blob + b"\x00" * (CHUNK_BYTES - len(blob))

        async def _prefill(pcm: np.ndarray, is_last: bool):
            """Run model.streaming_prefill on the executor."""
            nonlocal first_audio_seen
            msg_content: List[Any] = [pcm]
            # First-prefill carries the system message; modeling_minicpmo.py:1844-1869
            # expects the system role with a list-content shape on the very
            # first call for this session_id.
            if not first_audio_seen:
                # Send a leading system-only msg first (separately) so
                # streaming_prefill's `is_first` branch puts the system
                # prompt in front of <|im_start|>user. The upstream
                # tolerates a system-role prefill before user prefills.
                sp = system_prompt or _DEFAULT_SYSTEM_PROMPT
                def _sys():
                    with torch.no_grad():
                        # tokenizer/processor default to None — model
                        # lazy-loads via prepare_processor on first call.
                        model.streaming_prefill(
                            session_id=session_id,
                            msgs=[{"role": "system", "content": [sp]}],
                        )
                await loop.run_in_executor(_stream_executor, _sys)
                first_audio_seen = True

            def _user():
                with torch.no_grad():
                    model.streaming_prefill(
                        session_id=session_id,
                        msgs=[{"role": "user", "content": msg_content}],
                        is_last_chunk=is_last,
                    )
            await loop.run_in_executor(_stream_executor, _user)

        async def _generate_and_emit(max_new_tokens: int = 128, temperature: float = 0.6):
            """Run streaming_generate and push events back to the client."""
            nonlocal has_snapshot
            if temperature <= 0:
                await ws.send_json({
                    "event": "OmniError",
                    "error": "temperature must be > 0 (streaming_generate is sampling-only)",
                })
                return

            t0 = time.perf_counter()

            # Kick off the generator on the executor; iterate it chunk by
            # chunk through run_in_executor so abort_event is observed
            # between yields.
            def _make_gen():
                with torch.no_grad():
                    return iter(model.streaming_generate(
                        session_id=session_id,
                        do_sample=True,
                        max_new_tokens=max_new_tokens,
                        enable_speculative_snapshot=True,
                    ))
            gen = await loop.run_in_executor(_stream_executor, _make_gen)
            has_snapshot = True

            def _next():
                try:
                    return next(gen)
                except StopIteration:
                    return _SENTINEL

            full_text = ""
            while True:
                if abort_event.is_set():
                    break
                chunk = await loop.run_in_executor(_stream_executor, _next)
                if chunk is _SENTINEL:
                    break

                # streaming_generate yields (waveform_chunk, new_text) per
                # modeling_minicpmo.py:2370 / :2394 in the s3tokenizer path.
                waveform_chunk: Any = None
                new_text: str = ""
                if isinstance(chunk, tuple):
                    if len(chunk) >= 1:
                        waveform_chunk = chunk[0]
                    if len(chunk) >= 2 and isinstance(chunk[1], str):
                        new_text = chunk[1]
                elif isinstance(chunk, dict):
                    waveform_chunk = chunk.get("audio")
                    new_text = chunk.get("text") or ""
                else:
                    print(f"[ws-stream] unexpected chunk type "
                          f"{type(chunk).__name__}; skipping", flush=True)
                    continue

                if new_text:
                    cleaned = _strip_special(new_text)
                    if cleaned:
                        full_text += cleaned
                        try:
                            await ws.send_json({"type": "text_delta", "text": cleaned})
                        except Exception:
                            return

                if waveform_chunk is not None:
                    try:
                        if hasattr(waveform_chunk, "detach"):
                            arr = waveform_chunk.detach().cpu().numpy().astype("float32")
                        else:
                            arr = np.asarray(waveform_chunk, dtype="float32")
                        if arr.ndim > 1:
                            arr = arr.flatten()
                        if arr.size > 0:
                            try:
                                await ws.send_json({
                                    "type": "audio_b64",
                                    "data": _wav_to_b64(arr, 24000),
                                    "sample_rate": 24000,
                                })
                            except Exception:
                                return
                    except Exception as e:
                        print(f"[ws-stream] waveform encode failed: {e}", flush=True)

            try:
                if full_text:
                    await ws.send_json({
                        "type": "transcript", "text": full_text, "is_final": True,
                    })
                await ws.send_json({
                    "type": "done",
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "streaming_path": "ws-stream",
                    "aborted": abort_event.is_set(),
                })
            except Exception:
                pass

        async def _reset_for_next_turn(restore_snapshot: bool):
            """Called at every turn boundary so the next prefill builds on a
            clean state. `restore_snapshot=True` only on interrupt — for a
            naturally-completed turn the snapshot would just roll back to
            mid-generation state, then reset_session wipes it anyway, but
            empirically calling restore on every turn caused identical
            replies across turns (likely some state survives reset_session)."""
            nonlocal has_snapshot, any_audio_in_turn
            def _do():
                with torch.no_grad():
                    if restore_snapshot and has_snapshot:
                        try:
                            model.restore_speculative_snapshot()
                        except Exception as e:
                            print(f"[ws-stream] restore_speculative_snapshot "
                                  f"failed: {e}", flush=True)
                    try:
                        model.reset_session(reset_token2wav_cache=False)
                    except Exception as e:
                        print(f"[ws-stream] reset_session failed: {e}", flush=True)
            await loop.run_in_executor(_stream_executor, _do)
            has_snapshot = False
            pcm_accumulator.clear()
            any_audio_in_turn = False
            # Allow the NEXT turn's first prefill to re-emit the system
            # message (streaming_prefill's `is_first` branch fires when
            # session_id is None, which reset_session just made true).
            nonlocal first_audio_seen
            first_audio_seen = False

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                if msg.get("bytes") is not None:
                    blob: bytes = msg["bytes"]
                    if not blob:
                        continue
                    pcm_accumulator.extend(blob)
                    any_audio_in_turn = True
                    # Drain in 1-second blocks. Each drained block gets
                    # streaming_prefill(is_last_chunk=False) — more audio
                    # is coming. The trailing partial block stays in the
                    # accumulator until either another binary frame fills
                    # it OR a flush arrives.
                    while len(pcm_accumulator) >= CHUNK_BYTES:
                        block = bytes(pcm_accumulator[:CHUNK_BYTES])
                        del pcm_accumulator[:CHUNK_BYTES]
                        await _prefill(_bytes_to_pcm_f32(block), is_last=False)
                    continue

                if msg.get("text") is not None:
                    try:
                        ev = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    name = ev.get("event")
                    if name == "start":
                        if first_audio_seen:
                            try:
                                await ws.send_json({
                                    "event": "OmniError",
                                    "error": "'start' frame must precede any audio",
                                })
                            except Exception:
                                pass
                            await ws.close(code=1003)
                            return
                        system_prompt = ev.get("system_prompt") or None
                    elif name == "flush":
                        if not any_audio_in_turn:
                            try:
                                await ws.send_json({
                                    "event": "OmniError",
                                    "error": "flush before any audio in this turn",
                                })
                            except Exception:
                                pass
                            continue
                        # Send the trailing partial block (if any) with
                        # is_last_chunk=True. If the partial is smaller
                        # than the encoder's minimum, pad with silence.
                        # Two paths:
                        #   - len >= MIN_BYTES_FOR_PREFILL → pad up to
                        #     CHUNK_BYTES and send with is_last=True.
                        #   - len < MIN: drop it (already-flushed audio
                        #     is enough; one is_last=True call on a tail
                        #     of zeros is enough to close the turn).
                        tail = bytes(pcm_accumulator)
                        pcm_accumulator.clear()
                        if len(tail) < MIN_BYTES_FOR_PREFILL:
                            tail = b""   # rely on the most-recent committed chunk being last
                        if tail:
                            tail = _pad_to_chunk(tail)
                            await _prefill(_bytes_to_pcm_f32(tail), is_last=True)
                        else:
                            # Edge case: the user's whole utterance was a
                            # multiple of CHUNK_BYTES — last committed
                            # chunk went out with is_last=False. Send a
                            # silent CHUNK_BYTES with is_last=True to
                            # close the turn.
                            silence = b"\x00" * CHUNK_BYTES
                            await _prefill(_bytes_to_pcm_f32(silence), is_last=True)
                        max_new = int(ev.get("max_new_tokens", 128))
                        temp = float(ev.get("temperature", 0.6))
                        abort_event.clear()
                        await _generate_and_emit(max_new_tokens=max_new, temperature=temp)
                        # Prep state for the next turn on this same WS.
                        # Only restore the snapshot if the generation was
                        # aborted mid-stream — for naturally-completed
                        # turns, the snapshot would just point to a mid-
                        # generation state that reset_session is about to
                        # wipe anyway, AND empirically restoring on every
                        # turn boundary caused identical-text replies.
                        await _reset_for_next_turn(restore_snapshot=abort_event.is_set())
                        abort_event.clear()
                    elif name == "interrupt":
                        abort_event.set()
                        # The driver loop in _generate_and_emit will
                        # observe the event after the in-flight chunk
                        # completes; restore_and_reset is called from the
                        # flush handler's post-generate path. Here we just
                        # signal — if no generate is in flight, this is a
                        # no-op until the next flush.
                    elif name == "close":
                        break
                    # Unknown events silently ignored — forward-compat.
        except WebSocketDisconnect:
            pass
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ws-stream] handler crashed: {type(e).__name__}: {e}\n{tb}", flush=True)
            try:
                await ws.send_json({
                    "event": "OmniError",
                    "error": f"{type(e).__name__}: {e}",
                })
            except Exception:
                pass
        finally:
            # Always tear down the model session before releasing the lock.
            # On abrupt disconnect, restore the snapshot to roll back any
            # in-flight generation so the next session starts clean.
            try:
                await _reset_for_next_turn(restore_snapshot=True)
            except Exception as e:
                print(f"[ws-stream] cleanup failed: {e}", flush=True)
            try:
                await ws.close()
            except Exception:
                pass


