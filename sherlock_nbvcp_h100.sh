#!/bin/bash

#SBATCH --job-name=mcmc_nbvcp               # Job name
#SBATCH --partition=gpu,hns                 # Partitions with H100 GPUs
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --gpus-per-task=2                   # Request 2 GPUs per task
#SBATCH --constraint="GPU_SKU:H100_SXM5"    # Constraint for an H100 GPU
#SBATCH --cpus-per-gpu=2                    # Request 2 CPU cores per GPU. Adjust as needed.
#SBATCH --mem-per-gpu=80G                   # Request 80GB of host RAM per GPU. H100s have 80GB VRAM; adjust host RAM as needed.
#SBATCH --time=1-00:00:00                   # Time limit (D-HH:MM:SS), e.g., 1 day. Adjust based on expected runtime.

#SBATCH --output=logs/mcmc_nbvcp_job-%j.out  # Standard output log (%j expands to jobID)
#SBATCH --error=logs/mcmc_nbvcp_job-%j.err   # Standard error log

# Optional: Mail notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mrazo@stanford.edu 

# --- Safety and Setup ---
# Exit immediately if a command exits with a non-zero status.
set -e

# Ensure the logs directory exists
mkdir -p logs

echo "------------------------------------------------------------------------"
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "------------------------------------------------------------------------"

# --- Environment Setup ---
# Load necessary modules
module load python/3.12.1 cuda/12.6.1
module load math py-jax/0.4.36_py312 py-jaxlib/0.4.36_py312

# Create and activate a virtual environment
echo "Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Verify JAX can see the GPU (optional, but good for debugging)
echo "Checking JAX GPU availability..."
python3 -c "import jax; print(f'JAX Devices: {jax.devices()}')"

# Show GPU status
echo "Running nvidia-smi..."
nvidia-smi
echo "------------------------------------------------------------------------"

# --- Run the Python script ---
PYTHON_SCRIPT="./code/processing/10xGenomics/50-50_Jurkat-293T_mixture/mcmc_nbvcp_lognormal_mix_gpu.py"

echo "Running Python script: ${PYTHON_SCRIPT}"
python3 ${PYTHON_SCRIPT}

echo "------------------------------------------------------------------------"
echo "Python script finished with exit code $?."
echo "Job finished at $(date)"
echo "------------------------------------------------------------------------"