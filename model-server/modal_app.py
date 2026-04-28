"""Modal deploy of the trial-app model-server.

Use this when you don't have an AMD-ROCm machine handy. Modal hosts the
same /v1/transcribe?model=<id> API as model-server/server.py, running on
NVIDIA L4 (cheap, plenty for Parakeet + Canary).

Deploy:
    cd model-server
    modal deploy modal_app.py

That prints the public URL — set it on your local trial-app via:
    export MODEL_SERVER_URL=https://<workspace>--audio-trial-model-server-fastapi.modal.run

Then run the trial-app locally (uvicorn / pnpm dev) and the NeMo
adapters (parakeet, canary_1b_flash, canary_qwen_25b) talk to Modal
exactly the same way they would talk to the AMD docker container.

Why a separate file: AMD-ROCm and Modal-CUDA can't share one image
(rocm/pytorch base vs. python:3.10 + nemo wheels). Both call into the
same server.py + model_loader.py for the actual logic.
"""
from __future__ import annotations

import os

import modal

app = modal.App("audio-trial-model-server")

# Volume for HF + NeMo caches; survives image rebuilds.
VOL = modal.Volume.from_name("audio-trial-models", create_if_missing=True)

# Same approach as ambient-deploy/deployments/parakeet-asr/serve.py.
# Pinned to a working NeMo combo at 2026-04.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libsndfile1", "ffmpeg", "build-essential", "libsox-dev")
    .pip_install("Cython<3", "numpy<2")
    .pip_install(
        # NeMo 2.5.1's transitive dep nvidia-modelopt requires torch>=2.6.
        # 2.6 keeps Parakeet/Canary-1B working and exposes fully_shard at
        # the path NeMo 2.5 expects without our compatibility shim.
        "torch==2.6.0", "torchaudio==2.6.0",
        "pyarrow==15.0.2",
        # NeMo 2.4.0 brings the `canary2` prompt formatter (Canary-1B-flash)
        # and `nemo.collections.speechlm2` (Canary-Qwen-2.5B's SALM model).
        # 2.5+ ships the `qwen` prompt formatter that Canary-Qwen-2.5B's
        # cfg.prompt_format references. 2.4 errors out with "Unknown
        # prompt formatter: 'qwen'".
        "nemo_toolkit[all]==2.5.1",
        # speechlm2.SALM (Canary-Qwen) drags many training-time deps in
        # via its __init__ chain (seaborn, sacrebleu, lightning, …) — [all]
        # is the only extras combo that includes them all.
        "sacrebleu",
        "seaborn",
        "lightning>=2.4",
        "transformers>=4.45",
        "fastapi[standard]==0.115.4",
        "uvicorn[standard]==0.32.0",
        "soundfile",
        "huggingface_hub>=0.24,<1.0",
        "python-multipart",
        "httpx",
        "pydantic>=2.0",
    )
    .pip_install("numpy==1.26.4",
                 extra_options="--force-reinstall --no-deps")
    # Mount our server.py + model_loader.py into the image.
    .add_local_file("server.py", "/app/server.py")
    .add_local_file("model_loader.py", "/app/model_loader.py")
)


@app.cls(
    image=image,
    gpu="L4",                      # ample for Parakeet + Canary-1B; bump to A100 for Canary-Qwen
    volumes={"/models": VOL},
    timeout=3600,
    scaledown_window=600,           # stay warm 10 min
    min_containers=0,               # cold-start ok for testing
    max_containers=2,
    secrets=[
        # Optional — needed if the user adds gated models in Slice 2.
        # modal.Secret.from_name("hf-token"),
    ],
)
@modal.concurrent(max_inputs=4)
class ModelServer:
    @modal.enter()
    def _setup(self) -> None:
        # Persist HF + NeMo caches across cold-starts.
        os.environ.setdefault("HF_HOME", "/models/hf_cache")
        os.environ.setdefault("NEMO_CACHE_DIR", "/models/nemo_cache")
        # /app is the image-internal mount; ensure it's importable.
        import sys
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")

    @modal.asgi_app()
    def fastapi(self):
        # Reuse the same FastAPI app served on AMD.
        import sys
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        from server import app as server_app  # type: ignore
        return server_app
