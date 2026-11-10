# CLIMP experiments: Mamba-based CLIP (VMamba-B + Mamba text towers)

Reproduction, evaluation and extension of CLIMP
([Shabtay et al., arXiv 2601.06891](https://arxiv.org/abs/2601.06891)) on a SLURM cluster (L40S GPUs).

## Layout

| Path | What it does |
|---|---|
| `climp_model.py` | VMamba-B image tower, in-house Mamba-1 text tower, projection heads, COCO retrieval eval |
| `train_climp_ft.py`, `submit_ft.sh` | Original COCO fine-tune (VMamba-B + in-house 99M text tower) |
| `train_data.py` | Image-text datasets: COCO pairs; CC12M webdataset stream with alt-text / LLaVA-NeXT recaptions |
| `ablate/swap_model.py` | Any image tower x any Mamba text tower, frozen / partial / full training |
| `ablate/train_swap.py` | Contrastive training (DDP, resumable, streaming CC12M) |
| `ablate/chain.sh`, `ablate/cc12m.sh` | SLURM jobs: train then benchmark |
| `bench/` | CLIP-Benchmark evaluation (authors' protocol), efficiency, precision, geometry, qualitative |
| `bench/results/` | Result JSONs and the LaTeX results report |
| `data_prep/` | CC12M download + caption database |
| `tools/` | Small utilities |

## Setup

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install torch==2.5.1 transformers==4.46.3 numpy==1.26.4 webdataset safetensors torchvision
pip download --no-deps --no-binary :all: causal-conv1d==1.5.2 mamba-ssm==2.2.5 -d work/sdists
bash build_kernels.sh          # builds both from source (needs CUDA 12.1 + gcc-toolset-12)
git clone https://github.com/MzeroMiko/VMamba work/VMamba
```

Data, checkpoints and run outputs are not in the repository (about 1.2 TB).
CC12M: `python data_prep/cc12m_fetch.py` (pixparse/cc12m-wds + CaptionEmporium LLaVA-NeXT captions).

## Example

```bash
# tower-swap run: CLIMP-paper image tower (frozen) + public Mamba-790M text tower (trainable), COCO, 3 epochs
sbatch --job-name=demo ablate/chain.sh "demo paper hf:ckpt/hf/mamba-790m-hf 0 -1"

# CC12M training from ImageNet init, mixed captions, 2 GPUs
sbatch --job-name=cc12m --gres=gpu:2 --cpus-per-task=32 \
  --export=ALL,TAG=full_mix,VIS=inet,TRAIN_VIS=-1,TXT=hf:ckpt/hf/mamba-790m-hf,TRAIN_TXT=-1,CAP=mix,MAX_LEN=160,GC_TXT=1,VIS_AMP=1,EPOCHS=3 \
  ablate/cc12m.sh
```
