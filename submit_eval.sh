#!/bin/bash
#SBATCH --job-name=climp_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err
# Usage: sbatch submit_eval.sh [path/to/ckpt.pth]   (default: runs/v3_ft/climp_ft_best.pth)
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
cd /iitgn/home/syed.tashif/climp
venv/bin/python eval_climp.py --ckpt ${1:-runs/v3_ft/climp_ft_best.pth} --workers 14
