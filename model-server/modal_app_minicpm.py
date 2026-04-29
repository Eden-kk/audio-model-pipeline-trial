"""Modal deploy for MiniCPM-o-4.5 multimodal omni model.

This is the realtime-omni edge candidate (vs Gemini Live's cloud
candidate). Runs the model in HuggingFace Transformers via
`AutoModel.from_pretrained(..., trust_remote_code=True)` — NOT vLLM,
because upstream vLLM doesn't support MiniCPM-o-4.5; the only known
working vLLM path requires the `github.com/tc-mb/vllm` fork, which we're
deliberately avoiding to keep the deploy reproducible from PyPI alone.

The Transformers path is slower (no continuous batching, no PagedAttention)
but the trial-app is a bench tool with 1-3 concurrent users, so the
throughput hit is irrelevant. The TTFA hit (no streaming on first token)
is a known limitation for v1; it gets a "TTFA degraded" badge in the UI.

Why a separate file from `modal_app.py`:
  - GPU class differs: NeMo (Parakeet/Canary) runs fine on L4; MiniCPM-o
    needs A100-40GB minimum (model + TTS init = ~20-25 GB working set).
  - Image deps differ wildly: NeMo wants nemo_toolkit + Cython<3, this
    wants transformers + librosa + the model's auto-loaded code from HF.
  - Lifecycle: NeMo loads per-request via model_loader; this loads once
    in @modal.enter and keeps the model resident.

Deploy:
    cd model-server
    modal deploy modal_app_minicpm.py
    # → public URL printed; set MINICPM_O_REALTIME_URL in backend/.env

Smoke test:
    curl -X POST $MINICPM_O_REALTIME_URL/v1/omni \
        -H 'Content-Type: application/json' \
        -d '{"audio_b64": "<base64 16k mono PCM wav>",
             "system_prompt": "Reply briefly.",
             "generate_audio": false}'
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from typing import Any, Dict, List, Optional

import modal
from pydantic import BaseModel, Field


app = modal.App("audio-trial-minicpm-o")

# Sentinel used to detect generator exhaustion when running an iterator
# step on the executor pool (StopIteration can't cross thread boundaries
# cleanly).
_SENTINEL = object()


# ─── Request / response schemas (module scope) ──────────────────────────────
# Defined here, NOT inside MiniCPMOServer.fastapi(), so pydantic can fully
# resolve the type when FastAPI introspects the route signature. Local
# (closure-scope) BaseModel subclasses produce a ForwardRef pydantic v2
# can't `.rebuild()` reliably under FastAPI's auto-schema, which surfaces
# as 422 "missing query.req" + 500 on /openapi.json.

class OmniRequest(BaseModel):
    """One utterance worth of input. The trial-app's MiniCPMOAdapter sends
    one of these per user-turn under the chunked-HTTP fallback path."""
    audio_b64: "Optional[str]" = Field(  # forward-ref OK for stdlib types
        default=None,
        description="Base64 16 kHz mono PCM WAV bytes.",
    )
    image_b64: "Optional[str]" = Field(
        default=None,
        description="Base64 JPEG bytes. Single most-recent frame for v1.",
    )
    system_prompt: "Optional[str]" = Field(
        default=None,
        description="Optional system message prepended to the chat history.",
    )
    history: "Optional[list]" = Field(
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
    audio_b64: "Optional[str]" = None
    sample_rate: "Optional[int]" = None
    latency_ms: float
    model: str = "openbmb/MiniCPM-o-4_5"


# Stdlib forward refs above need typing imported at module level.
from typing import Optional   # noqa: E402  (placed here so the forward refs above resolve)
OmniRequest.model_rebuild()
OmniResponse.model_rebuild()

# Persistent volume for HF cache — model weights are ~18 GB, no point
# re-downloading on every cold start. Survives image rebuilds.
VOL = modal.Volume.from_name("audio-trial-minicpm-cache", create_if_missing=True)


# Image: torch 2.6 (matches modal_app.py's NeMo pin so a single-machine
# dev box doesn't conflict on CUDA libs); transformers >= 4.45 for the
# multimodal AutoProcessor; librosa for resampling reference audio. The
# model's repo code (auto-loaded via trust_remote_code) brings its own
# extras (decord, vector_quantize_pytorch, vocos, etc.) — pin the ones
# that are known-finicky here so a fresh image doesn't break.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libsndfile1", "ffmpeg", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        # MiniCPM-o-4.5's auto-loaded configuration_minicpmo.py imports
        # `Qwen3Config` which only exists from transformers 4.51.0+. The
        # ambient-deploy working setup pins this exact version — newer
        # transformers may also work but 4.51.0 is the known-good baseline.
        "transformers==4.51.0",
        "accelerate>=0.30",
        "soundfile",
        "librosa",
        "Pillow",
        "numpy<2",
        # OpenBMB's helper package — bundles the chat-template + audio
        # preprocessing utilities the model's auto-loaded code calls into.
        # Without it, init_tts() hits AttributeError on a missing util.
        "minicpmo-utils[all]>=1.0.5",
        # MiniCPM-o's auto-loaded code imports these unconditionally.
        "vector-quantize-pytorch",
        "vocos",
        "decord",
        "moviepy",
        # Web bits for the FastAPI surface.
        "fastapi[standard]==0.115.4",
        "uvicorn[standard]==0.32.0",
        "pydantic>=2.0",
        "huggingface_hub>=0.24,<1.0",
    )
)


# ─── FastAPI request/response shapes ────────────────────────────────────────

# Defined inline below using pydantic; declared here as module-scope so the
# ASGI app picks them up cleanly when Modal serializes the function.


@app.cls(
    image=image,
    gpu="A100-40GB",                   # MiniCPM-o-4.5 + TTS init ~ 20 GB
    volumes={"/root/.cache/huggingface": VOL},
    timeout=3600,                      # one-call ceiling; keeps long TTS in bounds
    scaledown_window=600,              # stay warm 10 min after the last call
    min_containers=0,                  # cold-start ok for a bench tool
    max_containers=2,                  # never auto-scale beyond two replicas
)
@modal.concurrent(max_inputs=2)        # two parallel inference calls per container
class MiniCPMOServer:
    @modal.enter()
    def _setup(self) -> None:
        """Load the model once when the container starts. Subsequent
        requests reuse the same in-memory model instance.

        Cold-start cost: ~2 min download (cached after first run via the
        volume) + ~30-60 s model + TTS init.
        """
        import torch
        from transformers import AutoModel  # noqa: WPS433

        print("[setup] Loading openbmb/MiniCPM-o-4_5 ...", flush=True)
        t0 = time.perf_counter()
        self._torch = torch  # cache the module so request handlers don't re-import

        # init_vision/audio/tts ALL true so the same container can serve
        # any of the realtime-omni request shapes without a reload. Each
        # init flag adds GPU memory + load time but keeps the API surface
        # uniform.
        self.model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-o-4_5",
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True,
        )
        self.model.eval().cuda()
        # init_tts() is the documented separate step required for audio
        # generation. Skipping it makes generate_audio=True silently
        # produce empty output.
        self.model.init_tts()

        load_s = time.perf_counter() - t0
        print(f"[setup] Ready in {load_s:.1f}s", flush=True)

    @modal.asgi_app()
    def fastapi(self):
        """ASGI entry — defines the public HTTP surface.

        OmniRequest / OmniResponse are module-scope pydantic models (above)
        so FastAPI's schema introspection sees fully-resolved types.
        """
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse

        torch = self._torch
        model = self.model

        api = FastAPI(title="MiniCPM-o realtime omni")

        # ── Shared input-parsing helpers ─────────────────────────────────
        # Used by both /v1/omni (blocking) and /v1/omni-stream (NDJSON).
        # Decoding audio + image is identical across endpoints; only the
        # generation step differs.

        def _build_msgs(req: OmniRequest) -> List[Dict[str, Any]]:
            import numpy as np  # noqa: WPS433
            import soundfile as sf  # noqa: WPS433
            from PIL import Image  # noqa: WPS433

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
            if not user_content:
                user_content.append("Hello.")

            msgs: List[Dict[str, Any]] = []
            if req.system_prompt:
                msgs.append({"role": "system", "content": [req.system_prompt]})
            if req.history:
                msgs.extend(req.history)
            msgs.append({"role": "user", "content": user_content})
            return msgs

        def _wav_to_b64(audio_arr, sr: int) -> str:
            """Encode a float32 numpy waveform → b64-encoded PCM-16 WAV."""
            import soundfile as sf  # noqa: WPS433
            buf = io.BytesIO()
            sf.write(buf, audio_arr, sr, format="WAV", subtype="PCM_16")
            return base64.b64encode(buf.getvalue()).decode("ascii")

        @api.get("/health")
        async def health() -> dict:
            return {"ok": True, "model": "openbmb/MiniCPM-o-4_5"}

        @api.post("/v1/omni", response_model=OmniResponse)
        async def omni(req: OmniRequest) -> OmniResponse:
            """One synchronous utterance → response.

            Decodes audio_b64 (and optional image_b64) into MiniCPM's
            native chat content list, calls model.chat(), and returns the
            text + (optionally) base64'd output WAV.
            """
            import numpy as np  # noqa: WPS433
            import soundfile as sf  # noqa: WPS433
            from PIL import Image  # noqa: WPS433

            t0 = time.perf_counter()

            # Build the user-turn content list. Order matters — MiniCPM-o
            # interprets the list left-to-right.
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
            if not user_content:
                # Pure-text turn (rare; mostly for warmup / smoke-test).
                user_content.append("Hello.")

            msgs: List[Dict[str, Any]] = []
            if req.system_prompt:
                msgs.append({"role": "system", "content": [req.system_prompt]})
            if req.history:
                msgs.extend(req.history)
            msgs.append({"role": "user", "content": user_content})

            # Decide TTS path. generate_audio=True triggers the slower TTS-
            # template path; False keeps the response text-only (fast).
            output_audio_path = None
            if req.generate_audio:
                # MiniCPM-o writes generated audio to disk (its API doesn't
                # expose an in-memory return). Use a per-request tmp file.
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                output_audio_path = tmp.name

            try:
                with torch.no_grad():
                    res = self.model.chat(
                        msgs=msgs,
                        do_sample=req.temperature > 0,
                        temperature=req.temperature,
                        max_new_tokens=req.max_new_tokens,
                        use_tts_template=req.generate_audio,
                        generate_audio=req.generate_audio,
                        output_audio_path=output_audio_path,
                    )
            except Exception as e:
                raise HTTPException(500, f"model.chat failed: {type(e).__name__}: {e}")

            # MiniCPM's chat() returns either a string (text-only) or a dict
            # ({text, audio_path, ...}) when TTS is on. Normalise.
            if isinstance(res, dict):
                text = res.get("text") or ""
            else:
                text = str(res)

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

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return OmniResponse(
                text=text,
                audio_b64=audio_b64_out,
                sample_rate=sample_rate,
                latency_ms=latency_ms,
            )

        @api.post("/v1/omni-stream")
        async def omni_stream(req: OmniRequest):
            """Streaming variant — emits NDJSON events as the model generates.

            Two execution paths, transparent to the client:
              1. **True streaming** via `model.streaming_chat(...)`. If the
                 loaded snapshot exposes a `streaming_chat` method that
                 yields `(text_chunk, audio_chunk_array, sr)` tuples, we
                 emit each chunk as it arrives — first audio byte in
                 ~500 ms.
              2. **Chunked-WAV fallback** when streaming_chat is missing
                 or errors out. We run blocking model.chat() to completion,
                 then split the resulting WAV into ~500 ms chunks and emit
                 them as separate audio_b64 events. Same client-side code
                 path; just slower TTFA.

            Either path emits the same NDJSON event schema:
              {"type":"transcript","text":"...","is_final":bool}
              {"type":"text_delta","text":"..."}
              {"type":"audio_b64","data":"...","sample_rate":int}
              {"type":"done","latency_ms":float,"streaming_path":"native"|"chunked"}
            """
            import json as _json
            import numpy as np  # noqa: WPS433
            import soundfile as sf  # noqa: WPS433

            t0 = time.perf_counter()
            msgs = _build_msgs(req)

            async def event_stream():
                # Tell the client which path was selected so the latency
                # dial can label "TTFA degraded" appropriately.
                streaming_path = "native"
                streaming_chat = getattr(model, "streaming_chat", None)

                # ── Path 1: native streaming_chat ────────────────────────
                if callable(streaming_chat) and req.generate_audio:
                    try:
                        full_text = ""
                        # MiniCPM's streaming_chat is sync; we run it in a
                        # thread so the FastAPI event loop stays responsive.
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
                        # Drain the generator from the executor pool too —
                        # each .__next__() may block on the model's TTS step.
                        while True:
                            try:
                                chunk = await loop.run_in_executor(None, lambda: next(gen, _SENTINEL))
                            except StopIteration:
                                break
                            if chunk is _SENTINEL:
                                break
                            # streaming_chat yield shape: per OpenBMB docs
                            #   (text_chunk: str, audio_arr: np.ndarray, sr: int)
                            # Be defensive about variations.
                            if isinstance(chunk, tuple) and len(chunk) == 3:
                                text_chunk, audio_chunk, sr = chunk
                            elif isinstance(chunk, dict):
                                text_chunk = chunk.get("text") or ""
                                audio_chunk = chunk.get("audio")
                                sr = chunk.get("sample_rate", 24000)
                            else:
                                # Unknown shape — give up native streaming
                                raise RuntimeError(
                                    f"streaming_chat yielded unexpected type {type(chunk).__name__}"
                                )
                            if text_chunk:
                                full_text += text_chunk
                                yield _json.dumps({
                                    "type": "text_delta", "text": text_chunk,
                                }) + "\n"
                            if audio_chunk is not None and len(audio_chunk) > 0:
                                yield _json.dumps({
                                    "type": "audio_b64",
                                    "data": _wav_to_b64(np.asarray(audio_chunk, dtype="float32"), int(sr)),
                                    "sample_rate": int(sr),
                                }) + "\n"
                        if full_text:
                            yield _json.dumps({
                                "type": "transcript",
                                "text": full_text, "is_final": True,
                            }) + "\n"
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                        yield _json.dumps({
                            "type": "done",
                            "latency_ms": latency_ms,
                            "streaming_path": streaming_path,
                        }) + "\n"
                        return
                    except Exception as e:
                        # streaming_chat exists but errored — surface a single
                        # text_delta noting we fell back, then continue to
                        # the chunked-fallback path below.
                        print(f"[omni-stream] streaming_chat failed, falling back: "
                              f"{type(e).__name__}: {e}", flush=True)
                        streaming_path = "chunked"

                # ── Path 2: chunked-WAV fallback ────────────────────────
                # Run blocking model.chat(), then chunk the resulting WAV
                # into ~500 ms slices.
                streaming_path = "chunked"
                output_audio_path = None
                if req.generate_audio:
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.close()
                    output_audio_path = tmp.name
                try:
                    # Run on the executor pool so the event loop stays free.
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
                        # Chunk into ~500 ms slices so the client can begin
                        # playback before the full WAV has been transferred.
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

                latency_ms = (time.perf_counter() - t0) * 1000.0
                yield _json.dumps({
                    "type": "done",
                    "latency_ms": latency_ms,
                    "streaming_path": streaming_path,
                }) + "\n"

            return StreamingResponse(
                event_stream(),
                media_type="application/x-ndjson",
            )

        return api
