# Long-term Memory — audio-model-pipeline-trial

Survives across sessions and `/compact` / `/clear`. Update at end of each session.

## Decisions made
- (2026-04-28) Initial Claude Code workspace set up. Dev box: `mlsys-b200.ucsd.edu`, Ubuntu 24.04. Workflow: ssh+tmux (no mosh; user lacks sudo). Login shell tcsh, but tmux runs bash.
- (2026-04-28, **corrected**) The dev box is **NVIDIA B200 × 8 (CUDA 12.9, driver 570)**, not AMD ROCm. Earlier note was wrong. Self-host runs via `MODEL_SERVER_BACKEND=local` against `model-server/run-cuda.sh` (native venv, no docker — see below). The AMD lane (`docker compose -f model-server/docker-compose.yml`) is preserved for other hosts.
- (2026-04-28) `yid042` is **not** in the host `docker` group and there is no rootless-docker / sudo path. Native venvs under `/raid/yid042/audio-stack/` are the way; `cloudflared` runs as a downloaded binary at `/raid/yid042/audio-stack/bin/cloudflared`. Avoid plans that assume `docker compose up`.
- (2026-04-28) Disk: `/` is 99% full (26 GB free); `/raid` has 9.8 TB. Anything weight-heavy goes to `/raid/yid042/...`. `MODEL_CACHE_DIR=/raid/yid042/audio-trial-models` is the canonical cache root for run-cuda.sh + run-vllm.sh.
- (2026-04-28) PyTorch on B200 needs **cu128** wheels (sm_100 kernels). cu126 wheels run torch 2.8 but raise `no kernel image is available for execution on the device`. Bootstrap script's TORCH_INDEX defaults to the cu128 channel.
- (2026-04-28) NeMo `[asr]` extras alone are **insufficient** — `modelPT.py` imports `nemo.lightning` which requires `megatron-core`, only available under `[all]`. Always pin `nemo_toolkit[all]==2.5.1` for the model-server.

## Failed approaches
- Tried `docker run --gpus all nvidia/cuda:12.4.1-base nvidia-smi` to validate the nvidia runtime — failed with "permission denied" on `/var/run/docker.sock` because user is not in the `docker` group. Don't repeat; pivot to native venvs immediately.

## Open questions
- (placeholder)
