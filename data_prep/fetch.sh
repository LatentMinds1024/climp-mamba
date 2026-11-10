#!/bin/bash
#SBATCH --job-name=cc12m_fetch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err
export PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60
cd /iitgn/home/syed.tashif/climp
venv/bin/python data_prep/cc12m_fetch.py
