#!/bin/bash
#SBATCH --job-name=climp_ft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err

# Usage:   sbatch submit_ft.sh                     (full run, auto-resumes from runs/v3_ft/climp_ft_latest.pth)
# Smoke:   sbatch --export=ALL,MAX_STEPS=60,EPOCHS=1,OUT_DIR=$HOME/climp/runs/smoke submit_ft.sh

export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export BATCH_PER_GPU=${BATCH_PER_GPU:-128}
NGPU=$(nvidia-smi -L | wc -l)
export NUM_WORKERS=${NUM_WORKERS:-$(( SLURM_CPUS_PER_TASK / NGPU - 1 ))}

cd /iitgn/home/syed.tashif/climp
echo "Host $(hostname)  GPUs $NGPU  workers/GPU $NUM_WORKERS  batch/GPU $BATCH_PER_GPU"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

venv/bin/torchrun --standalone --nproc_per_node=$NGPU train_climp_ft.py
