"""
Tower-swap contrastive training (controlled ablation + staged unfreezing).
Same data / schedule for every run so differences come from the towers only.

Env: VIS, TXT, TRAIN_VIS, TRAIN_TXT (see swap_model.py), EPOCHS, BATCH_PER_GPU, LR_*, OUT_DIR,
     INIT (optional ckpt from a previous stage to continue from -> staged unfreezing).
Launch: torchrun --nproc_per_node=2 ablate/train_swap.py
"""
import os, sys, json, math, time, datetime
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist, torch.distributed.nn as distnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swap_model import SwapCLIMP, ROOT
from climp_model import load_coco5k, retrieval_eval, train_tf
sys.path.insert(0, ROOT)
from train_data import PairDataset, cc12m_stream, cc12m_shards   # noqa: E402

E = os.environ.get
CFG = dict(VIS=E('VIS', 'paper'), TXT=E('TXT', 'inhouse'),
           TRAIN_VIS=int(E('TRAIN_VIS', 0)), TRAIN_TXT=int(E('TRAIN_TXT', -1)),
           DATA=E('DATA', 'coco'), EPOCHS=int(E('EPOCHS', 5)), BATCH_PER_GPU=int(E('BATCH_PER_GPU', 128)),
           LR_PROJ=float(E('LR_PROJ', 5e-4)), LR_TXT=float(E('LR_TXT', 3e-5)), LR_VIS=float(E('LR_VIS', 1e-5)),
           WD=0.05, WARMUP=int(E('WARMUP', 200)), MAX_LEN=int(E('MAX_LEN', 77)),
           OUT_DIR=E('OUT_DIR', f'{ROOT}/runs/swap/test'), INIT=E('INIT', ''),
           WORKERS=int(E('NUM_WORKERS', 8)), MAX_STEPS=int(E('MAX_STEPS', 0)), SEED=42, VIS_AMP=int(E('VIS_AMP', 0)),
           SHARDS=E('SHARDS', ''), CAP=E('CAP', 'alt'), EPOCH_SAMPLES=int(E('EPOCH_SAMPLES', 0)),
           EVAL_EVERY=int(E('EVAL_EVERY', 0)), SAVE_EVERY=int(E('SAVE_EVERY', 2000)), GC_TXT=int(E('GC_TXT', 0)))

DDP = 'RANK' in os.environ
if DDP:
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=60))
    RANK, WORLD, LOCAL = dist.get_rank(), dist.get_world_size(), int(os.environ['LOCAL_RANK'])
else:
    RANK, WORLD, LOCAL = 0, 1, 0
torch.cuda.set_device(LOCAL); DEV = torch.device('cuda', LOCAL); MAIN = RANK == 0
torch.manual_seed(CFG['SEED'] + RANK)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
os.makedirs(CFG['OUT_DIR'], exist_ok=True)
log = (lambda *a: print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)) if MAIN else (lambda *a: None)
log('Config', json.dumps(CFG))

model = SwapCLIMP(CFG['VIS'], CFG['TXT'], CFG['TRAIN_VIS'], CFG['TRAIN_TXT']).to(DEV)
model.vis_amp = bool(CFG['VIS_AMP'])
if CFG['GC_TXT'] and CFG['TRAIN_TXT'] != 0 and hasattr(model.text_enc.backbone, 'gradient_checkpointing_enable'):
    model.text_enc.backbone.gradient_checkpointing_enable()
if CFG['TRAIN_TXT'] == 0 and CFG['TXT'] != 'inhouse':
    model.text_enc.to(torch.bfloat16)
if CFG['INIT']:
    sd = torch.load(CFG['INIT'], map_location='cpu', weights_only=False)['model']
    from swap_model import load_partial
    load_partial(model, sd)
    log(f'INIT from {CFG["INIT"]}: loaded {len(sd)} tensors')
n_tr = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
log(f'vision {CFG["VIS"]}: trainable {n_tr(model.vis_enc):.1f}M / {sum(p.numel() for p in model.vis_enc.parameters())/1e6:.1f}M | '
    f'text {CFG["TXT"]}: trainable {n_tr(model.text_enc):.1f}M / {sum(p.numel() for p in model.text_enc.parameters())/1e6:.1f}M')

STREAM = CFG['DATA'] == 'cc12m'
def collate(b):
    ids, mask = model.tokenize([x[1] for x in b], CFG['MAX_LEN'])
    return torch.stack([x[0] for x in b]), ids, mask
if STREAM:
    shards = cc12m_shards(CFG['SHARDS'])
    ds = cc12m_stream(train_tf, shards, CFG['CAP'], seed=CFG['SEED'] + RANK)
    sampler = None
    n_samples = CFG['EPOCH_SAMPLES'] or len(shards) * 5030
    dl = DataLoader(ds, CFG['BATCH_PER_GPU'], num_workers=CFG['WORKERS'], pin_memory=True, drop_last=True,
                    persistent_workers=True, collate_fn=collate, prefetch_factor=4)
    steps_ep = n_samples // (CFG['BATCH_PER_GPU'] * WORLD)
    log(f'cc12m: {len(shards)} shards (~{n_samples:,} samples/epoch), captions={CFG["CAP"]}')
else:
    ds = PairDataset(CFG['DATA'], train_tf)
    sampler = DistributedSampler(ds, WORLD, RANK, shuffle=True, drop_last=True) if DDP else None
    dl = DataLoader(ds, CFG['BATCH_PER_GPU'], sampler=sampler, shuffle=sampler is None, num_workers=CFG['WORKERS'],
                    pin_memory=True, drop_last=True, persistent_workers=True, collate_fn=collate, prefetch_factor=4)
    steps_ep = min(len(dl), CFG['MAX_STEPS']) if CFG['MAX_STEPS'] else len(dl)
    n_samples = len(ds)
if CFG['MAX_STEPS']: steps_ep = min(steps_ep, CFG['MAX_STEPS'])
total = steps_ep * CFG['EPOCHS']
log(f'data {CFG["DATA"]}: {n_samples:,} pairs | {steps_ep} steps/epoch | global batch {CFG["BATCH_PER_GPU"]*WORLD}')

groups = [{'params': list(model.vis_proj.parameters()) + list(model.text_proj.parameters()) + [model.logit_scale], 'lr': CFG['LR_PROJ']},
          {'params': [p for p in model.vis_enc.parameters() if p.requires_grad], 'lr': CFG['LR_VIS']},
          {'params': [p for p in model.text_enc.parameters() if p.requires_grad], 'lr': CFG['LR_TXT']}]
groups = [g for g in groups if g['params']]
opt = torch.optim.AdamW(groups, weight_decay=CFG['WD'])
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / CFG['WARMUP'] if s < CFG['WARMUP'] else
                                          0.5 * (1 + math.cos(math.pi * min(1.0, (s - CFG['WARMUP']) / max(1, total - CFG['WARMUP'])))))
ddp = nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL]) if DDP else model
gather = (lambda x: torch.cat(distnn.all_gather(x), 0)) if DDP else (lambda x: x)


class _EvalTok:   # adapter for climp_model.retrieval_eval
    def __call__(self, caps, max_length=77, **kw):
        ids, mask = model.tokenize(caps, max_length)
        return {'input_ids': ids, 'attention_mask': mask}


val = load_coco5k(f'{ROOT}/data/coco')
hist = open(f'{CFG["OUT_DIR"]}/history.jsonl', 'a') if MAIN else None
STATE = f'{CFG["OUT_DIR"]}/state.pth'
step = 0
if os.path.exists(STATE):                                     # resume (preemption / wall-time)
    st = torch.load(STATE, map_location='cpu', weights_only=False)
    from swap_model import load_partial
    load_partial(model, st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    step = st['step']; log(f'RESUMED from {STATE} at step {step}')


def evaluate(tag):
    if not MAIN: return
    res = retrieval_eval(model, _EvalTok(), *val, DEV, max_len=77, num_workers=8)
    log(f'COCO val2017 5K [{tag}]', ' '.join(f'{k}={v:.2f}' for k, v in res.items() if not k.startswith('n_')))
    hist.write(json.dumps(dict(step=step, epoch=round(step / steps_ep, 3), **res)) + '\n'); hist.flush()
    model.train()


def save_state():
    if MAIN:
        torch.save(dict(model=model.trainable_state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                        swap_cfg=model.cfg, train_cfg=CFG), STATE + '.tmp')
        os.replace(STATE + '.tmp', STATE)
        torch.save(dict(model=model.trainable_state_dict(), swap_cfg=model.cfg, train_cfg=CFG, step=step),
                   f'{CFG["OUT_DIR"]}/last.pth')


it = iter(dl) if STREAM else None
while step < total:
    ep = step // steps_ep
    if not STREAM:
        if sampler: sampler.set_epoch(ep)
        it = iter(dl)
    model.train(); t0, tot, n = time.time(), 0.0, 0
    b = step % steps_ep
    while b < steps_ep:
        try:
            img, ids, mask = next(it)
        except StopIteration:
            break
        img, ids, mask = img.to(DEV, non_blocking=True), ids.to(DEV, non_blocking=True), mask.to(DEV, non_blocking=True)
        ie, te, scale = ddp(img, ids, mask)
        ai, at = gather(ie), gather(te)
        logits = scale * ai @ at.T
        lab = torch.arange(len(ai), device=DEV)
        loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); step += 1; b += 1
        with torch.no_grad(): model.logit_scale.clamp_(0, math.log(100))
        tot += loss.item(); n += 1
        if b % 100 == 0:
            log(f'ep {ep+1} [{b}/{steps_ep}] step {step}/{total} loss {tot/n:.4f} scale {scale.item():.1f} '
                f'mem {torch.cuda.max_memory_allocated(DEV)/2**30:.1f}G {(time.time()-t0)/n:.2f}s/it')
        if CFG['SAVE_EVERY'] and step % CFG['SAVE_EVERY'] == 0:
            save_state()
        if CFG['EVAL_EVERY'] and step % CFG['EVAL_EVERY'] == 0 and b < steps_ep:
            evaluate(f'step {step}')
        if DDP and ((CFG['SAVE_EVERY'] and step % CFG['SAVE_EVERY'] == 0) or (CFG['EVAL_EVERY'] and step % CFG['EVAL_EVERY'] == 0)):
            dist.barrier()
    log(f'=== epoch {ep+1} loss {tot/max(n,1):.4f} {(time.time()-t0)/60:.1f} min')
    evaluate(f'epoch {ep+1}'); save_state()
    if DDP: dist.barrier()
log('SWAP_DONE', CFG['OUT_DIR'])
if DDP: dist.destroy_process_group()
