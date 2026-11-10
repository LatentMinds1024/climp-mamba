#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
cd /iitgn/home/syed.tashif/climp
venv/bin/python bench/notes_eval.py --model $1 ${2:+--parts $2}
