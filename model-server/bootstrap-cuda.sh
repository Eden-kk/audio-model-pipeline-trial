#!/usr/bin/env bash
# Bootstrap a Python venv that can run the trial-app model-server natively on
# a NVIDIA CUDA box (B200 / H100 / A100). For environments where docker is
# unavailable (e.g. user not in `docker` group, no sudo).
#
# After this script succeeds you can launch the server with:
#   MODEL_CACHE_DIR=/path/to/cache bash model-server/run-cuda.sh
#
# Required:
#   uv installed and on PATH (https://astral.sh/uv).
# Optional env:
#   VENV_DIR  — destination venv (default /raid/yid042/audio-stack/venv-modelserver).
#   PYTHON_VERSION — default 3.10 (matches modal_app.py base image).
#   TORCH_INDEX — defaults to https://download.pytorch.org/whl/cu128.
#                 cu128 is required for B200 sm_100 kernels; cu126 / cu124
#                 wheels DO NOT have sm_100 and will fail with
#                 "no kernel image available for execution on the device".

set -euo pipefail

VENV_DIR="${VENV_DIR:-/raid/yid042/audio-stack/venv-modelserver}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install with:" >&2
  echo "  curl -fsSL https://astral.sh/uv/install.sh | sh" >&2
  exit 2
fi

uv python install "$PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

export VIRTUAL_ENV="$VENV_DIR"
export UV_LINK_MODE=copy   # /raid bind-mount may not support hardlinks

# 1. Torch + torchaudio + torchvision from the cu128 wheel index (sm_100).
uv pip install --python "$VENV_DIR/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url "$TORCH_INDEX" \
  "torch>=2.7" "torchaudio>=2.7" "torchvision>=0.22"

# 2. NeMo + the heavy stack. [all] (not [asr]) is required because
#    modelPT.py imports nemo.lightning, which transitively requires
#    megatron-core; megatron-core only ships under the [all] extras.
#    sacrebleu / seaborn / lightning / transformers are SALM (Canary-Qwen)
#    runtime deps that [all] pulls but we list explicitly to keep the
#    install order deterministic if upstream extras drift.
uv pip install --python "$VENV_DIR/bin/python" \
  "Cython<3" "numpy<2" "pyarrow==15.0.2" \
  "nemo_toolkit[all]==2.5.1" \
  sacrebleu seaborn "lightning>=2.4" "transformers>=4.45" \
  "fastapi[standard]==0.115.4" "uvicorn[standard]==0.32.0" \
  soundfile "huggingface_hub>=0.24,<1.0" python-multipart httpx "pydantic>=2.0" scipy

# 3. Pin numpy back to 1.26.4 — nvidia-modelopt (NeMo transitive dep) drags
#    in numpy 2.x which breaks np.float_ usage inside NeMo / lightning.
#    Same dance modal_app.py does at the bottom of its image build.
uv pip install --python "$VENV_DIR/bin/python" \
  "numpy==1.26.4" --force-reinstall --no-deps

echo
echo "model-server venv ready at $VENV_DIR"
echo "Smoke test:"
echo "  $VENV_DIR/bin/python -c 'import torch, nemo; print(torch.__version__, nemo.__version__, torch.cuda.is_available())'"
