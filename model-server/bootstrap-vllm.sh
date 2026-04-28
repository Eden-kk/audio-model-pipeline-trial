#!/usr/bin/env bash
# Bootstrap a Python venv that can run vLLM natively on a NVIDIA CUDA box
# (B200 / H100 / A100). Companion to bootstrap-cuda.sh; vLLM lives in its
# own venv because its torch pin would clash with NeMo's.
#
# After this script succeeds:
#   MODEL_CACHE_DIR=/path/to/cache bash model-server/run-vllm.sh
#
# Required:
#   uv installed and on PATH (https://astral.sh/uv).
# Optional env:
#   VENV_DIR  — destination venv (default /raid/yid042/audio-stack/venv-vllm).
#   PYTHON_VERSION — default 3.12 (vLLM supports 3.10–3.12 as of this writing;
#                    3.12 matches the system python on most modern boxes).
#   TORCH_INDEX — defaults to https://download.pytorch.org/whl/cu128 for B200
#                 sm_100 kernel coverage.

set -euo pipefail

VENV_DIR="${VENV_DIR:-/raid/yid042/audio-stack/venv-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install with:" >&2
  echo "  curl -fsSL https://astral.sh/uv/install.sh | sh" >&2
  exit 2
fi

uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"
export UV_LINK_MODE=copy

# vLLM ships its own torch pin via the install spec; we add the cu128 channel
# so the resolver can reach Blackwell-capable wheels. vLLM's setup.py picks
# the right CUDA flavor based on what torch is already installed.
#
# Pinning notes:
#   - vllm 0.11.0 is the latest version still built against the cu128 (CUDA
#     12.8) runtime. vllm 0.12+ requires libcudart.so.13 (CUDA 13), which
#     isn't packaged on PyPI for Blackwell yet.
#   - transformers<5 because vllm 0.11 reads tokenizer.all_special_tokens_extended,
#     which was removed in transformers 5.x. The 4.x line still has it.
uv pip install --python "$VENV_DIR/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url "$TORCH_INDEX" \
  "vllm==0.11.0" "transformers<5,>=4.49"

echo
echo "vllm venv ready at $VENV_DIR"
echo "Smoke test (lists OpenAI-compatible endpoints):"
echo "  $VENV_DIR/bin/vllm --help | head"
