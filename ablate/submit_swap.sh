#!/bin/bash
#SBATCH --job-name=swap
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err
# Usage: sbatch --job-name=T1 --export=ALL,VIS=paper,TXT=inhouse,TRAIN_TXT=-1,OUT_DIR=... ablate/submit_swap.sh
export PYTHONNOUSERSITE=1 CUDA_HOME=/usr/local/cuda-12.1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PATH=$CUDA_HOME/bin:$PATH
NGPU=$(nvidia-smi -L | wc -l)
export NUM_WORKERS=${NUM_WORKERS:-$(( SLURM_CPUS_PER_TASK / NGPU - 1 ))}
cd /iitgn/home/syed.tashif/climp
venv/bin/torchrun --standalone --nproc_per_node=$NGPU ablate/train_swap.py
