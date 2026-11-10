#!/bin/bash
#SBATCH --job-name=bench_eff_qual
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
cd /iitgn/home/syed.tashif/climp
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
venv/bin/python bench/qualitative.py
venv/bin/python bench/efficiency.py
echo EFFQUAL_DONE
