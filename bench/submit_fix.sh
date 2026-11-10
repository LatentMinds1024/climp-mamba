#!/bin/bash
#SBATCH --job-name=bench_fix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err
# Usage: sbatch bench/submit_fix.sh <model> [extra]   (extra = "eff" runs geometry calibration + efficiency first)
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
cd /iitgn/home/syed.tashif/climp
M=$1
R="venv/bin/python bench/run_bench.py --model $M --workers 14"
if [ "$2" = "eff" ]; then
  venv/bin/python bench/geometry_variants.py
  venv/bin/python bench/efficiency.py
fi
$R --task docci --res 224 --datasets urban1k
$R --task cb --res 224
$R --task cb --res 320 384
echo FIX_DONE
