# model-server — hosts NeMo ASR models on the AMD-ROCm remote machine.
#
# Base image: ROCm pytorch 6.2 (matches AMD's published wheels and the
# community parakeet_rocm fork). Override AMD_GFX_VERSION at build/run time
# for non-MI300 GPUs (e.g. consumer 7900 XTX needs gfx1100; MI300X = gfx942
# which is detected automatically).
FROM rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_2.4.0

# ── HSA / GFX overrides ──────────────────────────────────────────────────────
# MI300X / MI250X: leave HSA_OVERRIDE_GFX_VERSION unset.
# Consumer Navi (7900 XTX, etc): override to the matching gfx code.
ARG HSA_OVERRIDE_GFX_VERSION=""
ENV HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}

# ── system deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────────────────
# Note: NeMo on ROCm is currently a community fork. The requirements file
# pins the working combination at 2026-04 (see beecave-homelab/parakeet_rocm).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App ──────────────────────────────────────────────────────────────────────
COPY server.py model_loader.py ./

# Hugging Face cache: persist via volume mount in docker-compose.yml
ENV HF_HOME=/models/hf_cache
ENV NEMO_CACHE_DIR=/models/nemo_cache
ENV PYTHONUNBUFFERED=1

EXPOSE 9100

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9100"]
