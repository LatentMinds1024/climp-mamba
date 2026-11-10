#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=5-00:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err
# CC12M training (resumable: resubmit the same command to continue from OUT_DIR/state.pth), then benchmark.
# Usage: sbatch --job-name=TAG [--gres=gpu:2 --cpus-per-task=32] --export=ALL,TAG=...,VIS=..,TXT=..,CAP=.. ablate/cc12m.sh
export PYTHONNOUSERSITE=1 CUDA_HOME=/usr/local/cuda-12.1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PATH=$CUDA_HOME/bin:$PATH
cd /iitgn/home/syed.tashif/climp
NGPU=$(nvidia-smi -L | wc -l)
export DATA=cc12m OUT_DIR=runs/cc12m/$TAG NUM_WORKERS=${NUM_WORKERS:-$(( SLURM_CPUS_PER_TASK / NGPU - 2 ))}
mkdir -p $OUT_DIR
if [ ! -f $OUT_DIR/done ]; then
  venv/bin/torchrun --standalone --nproc_per_node=$NGPU ablate/train_swap.py && touch $OUT_DIR/done
fi
[ -f $OUT_DIR/done ] || exit 1
B="venv/bin/python bench/run_bench.py --model swap:$OUT_DIR/last.pth --tag cc12m_$TAG --workers 12"
$B --task cb --res 224 --datasets imagenet1k,flickr30k
$B --task hires --res 224 --datasets nocaps
$B --task docci --res 224 --datasets docci
echo "CC12M_DONE $TAG"
