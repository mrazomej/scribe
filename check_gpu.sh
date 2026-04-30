#!/usr/bin/env bash
set -euo pipefail

echo "Checking NVIDIA GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    if ! nvidia-smi; then
        echo "nvidia-smi failed; continuing so JAX diagnostics are still shown."
    fi
else
    echo "nvidia-smi not found. GPU might not be properly set up."
fi

echo "Configuring JAX environment (strict GPU mode)..."
# shellcheck disable=SC1091
if ! source scripts/setup_cuda_env.sh --gpu; then
    echo "GPU setup failed (non-fatal in check script)."
fi