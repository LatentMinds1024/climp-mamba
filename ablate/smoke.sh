#!/bin/bash
#SBATCH --job-name=swap_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err
export PYTHONNOUSERSITE=1 CUDA_HOME=/usr/local/cuda-12.1 TOKENIZERS_PARALLELISM=false NUM_WORKERS=8
cd /iitgn/home/syed.tashif/climp
for cfg in "paper inhouse 0 -1" "paper hf:ckpt/hf/mamba-130m-hf 0 -1" "paper paper 0 0" "inet paper -1 0"; do
  set -- $cfg
  VIS=$1 TXT=$2 TRAIN_VIS=$3 TRAIN_TXT=$4 EPOCHS=1 MAX_STEPS=30 OUT_DIR=runs/swap/smoke_$1_${2//[:\/]/_} \
    venv/bin/python ablate/train_swap.py 2>&1 | grep -vE "Warning|warn" | grep -E "trainable|data|ep 1|epoch|COCO|Error|error|Traceback|line" | head -12
done
