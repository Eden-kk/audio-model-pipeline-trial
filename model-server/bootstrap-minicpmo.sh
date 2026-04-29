#!/usr/bin/env bash
# Bootstrap a Python venv that can run MiniCPM-o-4.5 natively on a NVIDIA
# CUDA box (B200 / H100 / A100). Companion to bootstrap-cuda.sh and
# bootstrap-vllm.sh; MiniCPM lives in its OWN venv because transformers 4.51
# + minicpmo-utils[all] don't share well with NeMo's torch 2.11 + megatron
# stack OR vLLM 0.11's torch 2.8.
#
# After this script succeeds:
#   MODEL_CACHE_DIR=/path/to/cache bash model-server/run-minicpmo.sh
#
# Required:
#   uv installed and on PATH (https://astral.sh/uv).
# Optional env:
#   VENV_DIR  — destination venv (default /raid/yid042/audio-stack/venv-minicpmo).
#   PYTHON_VERSION — default 3.10. Don't bump to 3.12 — `decord`'s wheel
#                    coverage there is patchy.
#   TORCH_INDEX — defaults to https://download.pytorch.org/whl/cu128 for B200
#                 sm_100 kernel coverage.

set -euo pipefail

VENV_DIR="${VENV_DIR:-/raid/yid042/audio-stack/venv-minicpmo}"
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

# 1. torch + torchaudio + torchvision from cu128 (B200 sm_100 coverage).
#    Modal's modal_app_minicpm.py pins torch==2.6.0 — that's correct for
#    A100 but cu126 wheels lack sm_100. We use the cu128 channel and let
#    uv pick whichever 2.7+ wheel is current.
uv pip install --python "$VENV_DIR/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url "$TORCH_INDEX" \
  "torch>=2.7" "torchaudio>=2.7" "torchvision>=0.22" \
  torchcodec   # torchaudio 2.11 routes file I/O through torchcodec; without
               # it, MiniCPM-o's TTS step fails to load its prompt WAV
               # ("ImportError: TorchCodec is required for load_with_torchcodec").

# 2. The MiniCPM-o stack itself. Pinning notes:
#    - transformers==4.51.0 because the auto-loaded `configuration_minicpmo.py`
#      imports `Qwen3Config`, which only exists from 4.51+. Bumping to 4.52+
#      may also work but 4.51.0 is the known-good baseline (matches Modal's pin).
#    - minicpmo-utils[all] bundles the chat-template + audio preprocessing
#      utilities the model's auto-loaded code calls into. Without it,
#      init_tts() hits AttributeError on a missing util.
#    - vector-quantize-pytorch / vocos / decord / moviepy are imported
#      unconditionally by the auto-loaded code; pin explicitly so a fresh
#      install doesn't break.
uv pip install --python "$VENV_DIR/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url "$TORCH_INDEX" \
  "transformers==4.51.0" "accelerate>=0.30" \
  soundfile librosa Pillow "numpy<2" \
  "minicpmo-utils[all]>=1.0.5" \
  vector-quantize-pytorch vocos decord moviepy \
  "fastapi[standard]==0.115.4" "uvicorn[standard]==0.32.0" \
  "pydantic>=2.0" "huggingface_hub>=0.24,<1.0"

# 3. Pin numpy to 1.26.4 — the auto-loaded MiniCPM code uses np.float_ which
#    numpy 2.x removed. Same dance bootstrap-cuda.sh + the Modal app's
#    final pip step both do.
uv pip install --python "$VENV_DIR/bin/python" \
  "numpy==1.26.4" --force-reinstall --no-deps

echo
echo "minicpmo venv ready at $VENV_DIR"
echo "Smoke test:"
echo "  $VENV_DIR/bin/python -c 'from transformers.models.qwen3 import Qwen3Config; \\"
echo "    import torch; print(torch.__version__, torch.cuda.is_available(), \\"
echo "    torch.cuda.get_device_capability())'"
