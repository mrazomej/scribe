#!/bin/bash
echo "Checking NVIDIA GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "Testing JAX GPU access..."
    python3 -c "import jax; print('Number of GPUs available:', len(jax.devices('gpu')))"
else
    echo "nvidia-smi not found. GPU might not be properly set up."
fi