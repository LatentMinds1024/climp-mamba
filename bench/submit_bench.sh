#!/bin/bash
#SBATCH --job-name=climp_bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err
# Usage: sbatch --job-name=bench_paper bench/submit_bench.sh paper [pre-command]
# Runs the whole CLIMP-paper benchmark suite for one model; results are per-file and resumable.
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
cd /iitgn/home/syed.tashif/climp
M=$1
R="venv/bin/python bench/run_bench.py --model $M --workers 14"
[ -n "$2" ] && eval "$2"
$R --task hires --res 224 --datasets nocaps          # sanity check vs authors' README
$R --task cb    --res 224                            # Tables 1, 2, 9, 10
$R --task hires --res 224 512 896                    # Tables 4, 11
$R --task docci --res 896 224                        # Tables 5, 13
$R --task geometry --res 224                         # Table 6
$R --task cb    --res 320 384                        # Table 3
echo BENCH_DONE
