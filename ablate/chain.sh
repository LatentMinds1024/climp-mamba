#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/iitgn/home/syed.tashif/climp/logs/%x_%j.out
#SBATCH --error=/iitgn/home/syed.tashif/climp/logs/%x_%j.err
# Usage: sbatch --job-name=chainA ablate/chain.sh "TAG VIS TXT TRAIN_VIS TRAIN_TXT" ...
# Each run: 3 epochs COCO train2017 (1 GPU, batch 128), then benchmark on the same suite.
export PYTHONNOUSERSITE=1 CUDA_HOME=/usr/local/cuda-12.1 TOKENIZERS_PARALLELISM=false NUM_WORKERS=12
export PATH=$CUDA_HOME/bin:$PATH
cd /iitgn/home/syed.tashif/climp
for cfg in "$@"; do
  set -- $cfg; TAG=$1; OUT=runs/swap/$TAG
  if [ ! -f $OUT/done ]; then
    rm -f $OUT/history.jsonl
    VIS=$2 TXT=$3 TRAIN_VIS=$4 TRAIN_TXT=$5 EPOCHS=${EPOCHS:-3} OUT_DIR=$OUT \
      venv/bin/python ablate/train_swap.py && touch $OUT/done
  fi
  B="venv/bin/python bench/run_bench.py --model swap:$OUT/last.pth --tag swap_$TAG --workers 12"
  $B --task cb --res 224 --datasets imagenet1k,flickr30k
  $B --task hires --res 224 --datasets nocaps
  $B --task docci --res 224 --datasets docci
  echo "CHAIN_RUN_DONE $TAG"
done
