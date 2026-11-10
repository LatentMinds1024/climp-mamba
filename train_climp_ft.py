"""
CLIMP v3 -- full fine-tuning (VMamba-B vision tower + in-house Mamba-1 text tower).
Port of kagglefiles/climp_v3_finetune_2gpu.ipynb to the IITGN SLURM cluster (L40S GPUs).

Launch (inside SLURM):  torchrun --nproc_per_node=$NGPU train_climp_ft.py
Everything is configurable through environment variables (see CONFIG below).
"""
import os, sys, json, math, time, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as distnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

# ───────────────────────────── CONFIG ─────────────────────────────
ROOT = os.path.expanduser('~/climp')
E = os.environ.get
CFG = dict(
    TEXT_CKPT   = E('TEXT_CKPT',  f'{ROOT}/ckpt/ckpt/best_model.pt'),          # in-house Mamba-1
    TEXT_CFG    = E('TEXT_CFG',   f'{ROOT}/ckpt/mamba-final/config.json'),
    TOKENIZER   = E('TOKENIZER',  f'{ROOT}/ckpt/mamba-final/tokenizer.json'),
    VMAMBA_DIR  = E('VMAMBA_DIR', f'{ROOT}/work/VMamba/classification/models'),
    VMAMBA_CKPT = E('VMAMBA_CKPT', f'{ROOT}/ckpt/vssm_base_ckpt.pth'),
    COCO        = E('COCO',       f'{ROOT}/data/coco'),
    OUT_DIR     = E('OUT_DIR',    f'{ROOT}/runs/v3_ft'),
    FROZEN_CKPT = E('FROZEN_CKPT', ''),          # optional v2 projection-head warm start
    EPOCHS      = int(E('EPOCHS', 10)),
    BATCH_PER_GPU = int(E('BATCH_PER_GPU', 64)),
    ACCUM_STEPS = int(E('ACCUM_STEPS', 1)),
    EMBED_DIM   = 512,
    MAX_SEQ_LEN = int(E('MAX_SEQ_LEN', 77)),
    LR_PROJ     = float(E('LR_PROJ', 5e-4)),
    LR_TEXT     = float(E('LR_TEXT', 1e-5)),
    LR_VIS      = float(E('LR_VIS',  5e-6)),
    WEIGHT_DECAY= float(E('WEIGHT_DECAY', 0.01)),
    WARMUP_STEPS= int(E('WARMUP_STEPS', 200)),
    GRAD_CKPT   = int(E('GRAD_CKPT', 1)),        # gradient checkpointing inside VMamba
    VIS_AMP     = int(E('VIS_AMP', 0)),          # 0 = VMamba fp32 (as in notebook), 1 = bf16 autocast
    NUM_WORKERS = int(E('NUM_WORKERS', 8)),
    MAX_STEPS   = int(E('MAX_STEPS', 0)),        # >0: stop each epoch after N micro-steps (smoke test)
    EVAL_EVERY  = int(E('EVAL_EVERY', 1)),       # evaluate on COCO 5K every N epochs
    SAVE_EVERY_STEPS = int(E('SAVE_EVERY_STEPS', 1000)),
    SEED        = int(E('SEED', 42)),
)

# ───────────────────────────── DDP setup ─────────────────────────────
DDP = 'RANK' in os.environ
if DDP:
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=60))
    RANK, WORLD, LOCAL = dist.get_rank(), dist.get_world_size(), int(os.environ['LOCAL_RANK'])
else:
    RANK, WORLD, LOCAL = 0, 1, 0
torch.cuda.set_device(LOCAL)
DEVICE = torch.device('cuda', LOCAL)
MAIN = RANK == 0
torch.manual_seed(CFG['SEED'] + RANK)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.makedirs(CFG['OUT_DIR'], exist_ok=True)

def log(*a):
    if MAIN:
        print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)

log('Config:', json.dumps(CFG, indent=1))
log(f'World size {WORLD} x {torch.cuda.get_device_name(LOCAL)}')

from climp_model import (VisionMambaEncoder, TextMambaEncoder, CLIMP, build_tokenizer,
                         train_tf, load_coco5k, retrieval_eval)

tokenizer = build_tokenizer(CFG['TOKENIZER'])

def tokenize(caps):
    t = tokenizer(caps, max_length=CFG['MAX_SEQ_LEN'], padding='max_length',
                  truncation=True, return_tensors='pt')
    return t['input_ids'], t['attention_mask']

class COCOPairDataset(Dataset):
    """One image per item, one of its 5 captions sampled at random."""
    def __init__(self, img_dir, ann_file):
        data = json.load(open(ann_file))
        id2file = {x['id']: x['file_name'] for x in data['images']}
        caps = {}
        for a in data['annotations']:
            caps.setdefault(a['image_id'], []).append(a['caption'].strip())
        self.pairs = [(id2file[i], c) for i, c in caps.items()]
        self.img_dir = img_dir
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        f, caps = self.pairs[idx]
        cap = caps[torch.randint(len(caps), (1,)).item()]
        img = train_tf(Image.open(os.path.join(self.img_dir, f)).convert('RGB'))
        return img, cap

def collate_train(b):
    imgs = torch.stack([x[0] for x in b]); ids, mask = tokenize([x[1] for x in b])
    return imgs, ids, mask

coco = CFG['COCO']
train_ds = COCOPairDataset(f'{coco}/train2017', f'{coco}/annotations/captions_train2017.json')
sampler  = DistributedSampler(train_ds, WORLD, RANK, shuffle=True, drop_last=True) if DDP else None
train_loader = DataLoader(train_ds, batch_size=CFG['BATCH_PER_GPU'], sampler=sampler,
                          shuffle=sampler is None, num_workers=CFG['NUM_WORKERS'],
                          pin_memory=True, drop_last=True, persistent_workers=True,
                          collate_fn=collate_train, prefetch_factor=4)

val_set = load_coco5k(coco)   # (img_dir, files, captions, cap2img)
log(f'Train: {len(train_ds):,} images | {len(train_loader)} steps/epoch/GPU | '
    f'global batch {CFG["BATCH_PER_GPU"] * WORLD} (x{CFG["ACCUM_STEPS"]} accum)')
log(f'Val  : COCO 5K {len(val_set[1])} images, {len(val_set[2])} captions')

# ───────────────────────────── Model ─────────────────────────────
vis_enc  = VisionMambaEncoder(CFG['VMAMBA_CKPT'], CFG['GRAD_CKPT'], log=log)
text_enc = TextMambaEncoder(CFG['TEXT_CFG'], CFG['TEXT_CKPT'], log=log)
model    = CLIMP(vis_enc, text_enc, CFG['EMBED_DIM'], vis_amp=CFG['VIS_AMP']).to(DEVICE)

if CFG['FROZEN_CKPT'] and os.path.exists(CFG['FROZEN_CKPT']):
    fz = torch.load(CFG['FROZEN_CKPT'], map_location='cpu', weights_only=False)
    model.vis_proj.load_state_dict(fz['vis_proj']); model.text_proj.load_state_dict(fz['text_proj'])
    model.logit_scale.data = fz['logit_scale'].to(DEVICE)
    log('Projection heads warm-started from', CFG['FROZEN_CKPT'])
else:
    log('Projection heads initialised from scratch')

n_vis = sum(p.numel() for p in model.vis_enc.parameters())
n_txt = sum(p.numel() for p in model.text_enc.parameters())
n_all = sum(p.numel() for p in model.parameters())
log(f'Params: vision {n_vis/1e6:.1f}M | text {n_txt/1e6:.1f}M | total {n_all/1e6:.1f}M (all trainable)')

param_groups = [
    {'params': list(model.vis_proj.parameters()),  'lr': CFG['LR_PROJ'], 'name': 'vis_proj'},
    {'params': list(model.text_proj.parameters()), 'lr': CFG['LR_PROJ'], 'name': 'txt_proj'},
    {'params': [model.logit_scale],                'lr': CFG['LR_PROJ'], 'name': 'scale', 'weight_decay': 0.0},
    {'params': list(model.vis_enc.parameters()),   'lr': CFG['LR_VIS'],  'name': 'vis_enc'},
    {'params': list(model.text_enc.parameters()),  'lr': CFG['LR_TEXT'], 'name': 'txt_enc'},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=CFG['WEIGHT_DECAY'])

steps_per_epoch = len(train_loader) if not CFG['MAX_STEPS'] else min(CFG['MAX_STEPS'], len(train_loader))
total_steps = max(1, steps_per_epoch * CFG['EPOCHS'] // CFG['ACCUM_STEPS'])
def lr_lambda(step):
    if step < CFG['WARMUP_STEPS']:
        return step / max(1, CFG['WARMUP_STEPS'])
    p = (step - CFG['WARMUP_STEPS']) / max(1, total_steps - CFG['WARMUP_STEPS'])
    return max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL]) if DDP else model

# ───────────────────────────── Checkpointing ─────────────────────────────
CKPT_LATEST = f'{CFG["OUT_DIR"]}/climp_ft_latest.pth'
CKPT_BEST   = f'{CFG["OUT_DIR"]}/climp_ft_best.pth'

def save_ckpt(path, epoch, step, batch_in_epoch, best_rsum, extra=None):
    if not MAIN: return
    tmp = path + '.tmp'
    torch.save(dict(epoch=epoch, step=step, batch_in_epoch=batch_in_epoch, best_rsum=best_rsum,
                    model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), cfg=CFG, **(extra or {})), tmp)
    os.replace(tmp, path)

start_epoch, global_step, skip_batches, best_rsum = 0, 0, 0, -1.0
if os.path.exists(CKPT_LATEST):
    s = torch.load(CKPT_LATEST, map_location='cpu', weights_only=False)
    model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
    scheduler.load_state_dict(s['scheduler'])
    start_epoch, global_step, best_rsum = s['epoch'], s['step'], s['best_rsum']
    skip_batches = s['batch_in_epoch']
    if skip_batches >= steps_per_epoch:
        start_epoch, skip_batches = start_epoch + 1, 0
    log(f'Resumed from {CKPT_LATEST}: epoch {start_epoch+1}, batch {skip_batches}, '
        f'step {global_step}, best Rsum {best_rsum:.2f}')
    del s

# ───────────────────────────── Evaluation (rank 0) ─────────────────────────────
def evaluate():
    return retrieval_eval(model, tokenizer, *val_set, DEVICE, max_len=CFG['MAX_SEQ_LEN'],
                          ks=(1, 5, 10), num_workers=CFG['NUM_WORKERS'])

def print_eval(res, tag):
    log(f'COCO 5K [{tag}]  ' + '  '.join(f'{k}={v:.2f}' for k, v in res.items() if not k.startswith('n_')) +
        f'  (v2 frozen baseline Rsum=139.87, delta {res["Rsum"]-139.87:+.2f})')

# ───────────────────────────── Training ─────────────────────────────
def gather(x):
    return torch.cat(distnn.all_gather(x), 0) if DDP else x   # keeps gradients

log(f'Optimizer steps: {total_steps} | warmup {CFG["WARMUP_STEPS"]} | start epoch {start_epoch+1}/{CFG["EPOCHS"]}')
hist_path = f'{CFG["OUT_DIR"]}/history.jsonl'
model.train()
for epoch in range(start_epoch, CFG['EPOCHS']):
    if sampler: sampler.set_epoch(epoch)
    t0, run_loss, n, gnorm = time.time(), 0.0, 0, torch.tensor(0.0)
    optimizer.zero_grad(set_to_none=True)
    for b, (images, ids, mask) in enumerate(train_loader):
        if b >= steps_per_epoch: break
        if b < skip_batches: continue           # fast-forward after mid-epoch resume
        images = images.to(DEVICE, non_blocking=True)
        ids, mask = ids.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True)

        sync = (b + 1) % CFG['ACCUM_STEPS'] == 0
        ctx = ddp_model.no_sync() if (DDP and not sync) else torch.enable_grad()
        with ctx:
            img_emb, txt_emb, scale = ddp_model(images, ids, mask)
            # all-gather so every GPU contrasts against the whole global batch
            all_img, all_txt = gather(img_emb), gather(txt_emb)
            logits = scale * all_img @ all_txt.T
            labels = torch.arange(len(all_img), device=DEVICE)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            # all_gather backward sums grads over ranks and DDP averages them -> exact global grad
            (loss / CFG['ACCUM_STEPS']).backward()

        if sync:
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            global_step += 1
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))

        run_loss += loss.item(); n += 1
        if MAIN and (b + 1) % 50 == 0:
            el = time.time() - t0
            eta = el / n * (steps_per_epoch - b - 1)
            log(f'ep {epoch+1} [{b+1}/{steps_per_epoch}] loss={run_loss/n:.4f} '
                f'scale={scale.item():.1f} gnorm={gnorm:.2f} lr_vis={optimizer.param_groups[3]["lr"]:.2e} '
                f'mem={torch.cuda.max_memory_allocated(DEVICE)/2**30:.1f}G '
                f'{el/n:.2f}s/it eta={eta/60:.0f}min')
        if CFG['SAVE_EVERY_STEPS'] and (b + 1) % CFG['SAVE_EVERY_STEPS'] == 0 and sync:
            save_ckpt(CKPT_LATEST, epoch, global_step, b + 1, best_rsum)
    skip_batches = 0

    avg = run_loss / max(n, 1)
    log(f'=== Epoch {epoch+1}/{CFG["EPOCHS"]} done | loss={avg:.4f} | {(time.time()-t0)/60:.1f} min')
    record = dict(epoch=epoch + 1, step=global_step, train_loss=avg)
    if MAIN and ((epoch + 1) % CFG['EVAL_EVERY'] == 0 or epoch + 1 == CFG['EPOCHS']):
        res = evaluate(); print_eval(res, f'epoch {epoch+1}'); record.update(res)
        if res['Rsum'] > best_rsum:
            best_rsum = res['Rsum']
            save_ckpt(CKPT_BEST, epoch, global_step, steps_per_epoch, best_rsum, {'eval': res})
            log(f'New best Rsum {best_rsum:.2f} -> {CKPT_BEST}')
    save_ckpt(CKPT_LATEST, epoch, global_step, steps_per_epoch, best_rsum)
    if MAIN:
        with open(hist_path, 'a') as f: f.write(json.dumps(record) + '\n')
    if DDP: dist.barrier()

log(f'Done. Best Rsum {best_rsum:.2f}  best ckpt: {CKPT_BEST}')
if DDP: dist.destroy_process_group()
