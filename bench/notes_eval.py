"""
Evaluation exactly as in the user's notes, for CLIP (pt) / CLIMP (pt) / CLIMP (ours).

Page 1 (accuracy):  COCO 5K + NoCaps retrieval R@1/3/5/10 (both directions), ImageNet-1k zero-shot
                    Acc@1/Acc@5, ImageNet Acc vs resolution.
Page 2 (efficiency, image encoder, x5 repeats):  latency, peak VRAM, energy (J/img), GFLOPs,
                    throughput at 64..1024 px;  accuracy drop FP32 -> BF16 -> INT8 -> INT4.

Precision: FP32 = no autocast; BF16 = torch.autocast(bfloat16); INT8 / INT4 = weight-only
round-to-nearest symmetric quantisation of every Linear/Conv weight (INT8 per output channel,
INT4 groups of 128), embeddings and norms kept FP32 (W8A32 / W4A32 simulated).

Usage: python bench/notes_eval.py --model clip|paper|ours [--parts acc,res,prec,eff]
"""
import os, sys, json, time, copy, argparse, statistics
from contextlib import nullcontext
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load_model, square_transform, ROOT, MEAN, STD
from run_bench import CaptionDS, nocaps_items, wds_dir, wds_workers
from clip_benchmark.datasets.builder import build_dataset, image_captions_collate_fn
from clip_benchmark.metrics import zeroshot_classification as zsc

OUT_DIR = f'{ROOT}/bench/results/notes'
os.makedirs(OUT_DIR, exist_ok=True)
dev = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = False      # true FP32 numbers
torch.backends.cudnn.allow_tf32 = False


def coco5k_items():
    v = json.load(open(f'{ROOT}/data/coco/annotations/captions_val2017.json'))
    caps = {}
    for a in v['annotations']:
        caps.setdefault(a['image_id'], []).append(a['caption'].strip())
    id2f = {x['id']: x['file_name'] for x in v['images']}
    return [(f'{ROOT}/data/coco/val2017/{id2f[i]}', caps[i]) for i in sorted(caps)]


def tf(model, res):
    return square_transform(res, getattr(model, 'mean', MEAN), getattr(model, 'std', STD))


def prec_ctx(prec):
    return torch.autocast('cuda', dtype=torch.bfloat16) if prec == 'bf16' else nullcontext()


# ───────────────────────────── quantisation (weight-only, simulated) ─────────────────────────────
@torch.no_grad()
def fake_quant_(model, bits, group=128):
    n = 0
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d)) and mod.weight.numel() >= 1024:
            W = mod.weight.data
            W2 = W.reshape(W.shape[0], -1).float()
            if bits == 8:
                s = W2.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
                Wq = (W2 / s).round().clamp(-127, 127) * s
            else:
                cols = W2.shape[1]; pad = (-cols) % group
                Wp = F.pad(W2, (0, pad)).reshape(W2.shape[0], -1, group)
                s = Wp.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7
                Wq = ((Wp / s).round().clamp(-8, 7) * s).reshape(W2.shape[0], -1)[:, :cols]
            mod.weight.data = Wq.reshape(W.shape).to(W.dtype); n += 1
    return n


# ───────────────────────────── accuracy ─────────────────────────────
@torch.no_grad()
def retrieval(model, items, res, prec, ks=(1, 3, 5, 10), bs=64):
    dl = DataLoader(CaptionDS(items, tf(model, res)), batch_size=bs, num_workers=12, pin_memory=True,
                    collate_fn=image_captions_collate_fn)
    I, T, t2i = [], [], []
    for imgs, caps in dl:
        with prec_ctx(prec):
            I.append(F.normalize(model.encode_image(imgs.to(dev)).float(), dim=-1))
            flat = [c for cs in caps for c in cs]
            T.append(F.normalize(model.encode_text(model.tokenizer(flat).to(dev)).float(), dim=-1))
        base = sum(len(x) for x in I[:-1])
        t2i += [base + i for i, cs in enumerate(caps) for _ in cs]
    I, T, t2i = torch.cat(I), torch.cat(T), torch.tensor(t2i, device=dev)
    S = T @ I.T                                                      # (captions, images)
    K = max(ks)
    t2i_hit = S.topk(K, 1).indices == t2i[:, None]
    i2t_hit = t2i[S.T.topk(K, 1).indices] == torch.arange(len(I), device=dev)[:, None]
    r = {}
    for k in ks:
        r[f'T2I_R@{k}'] = t2i_hit[:, :k].any(1).float().mean().item() * 100
        r[f'I2T_R@{k}'] = i2t_hit[:, :k].any(1).float().mean().item() * 100
    r['n_images'], r['n_captions'] = len(I), len(T)
    return r


@torch.no_grad()
def imagenet(model, res, prec, bs=128):
    ds = build_dataset('wds/imagenet1k', root=wds_dir('imagenet1k'), transform=tf(model, res), split='test')
    dl = DataLoader(ds, batch_size=bs, num_workers=wds_workers('imagenet1k', 14), pin_memory=True)
    with prec_ctx(prec):
        m = zsc.evaluate(model, dl, model.tokenizer, ds.classes, ds.templates, 'cuda', amp=False)
    return {'Acc@1': m['acc1'] * 100, 'Acc@5': m['acc5'] * 100}


# ───────────────────────────── efficiency (image encoder) ─────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit(); _H = pynvml.nvmlDeviceGetHandleByIndex(0)
    energy_mj = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(_H)
except Exception:
    energy_mj = None


class ImageEncoder(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m.encode_image(x)


def gflops(model, res):
    from fvcore.nn import flop_count
    ops = {}
    try:
        from functools import partial
        from csms6s import selective_scan_flop_jit
        ops = {'prim::PythonOp.SelectiveScanCuda': partial(selective_scan_flop_jit, backend='prefixsum', verbose=False),
               'aten::silu': None, 'aten::neg': None, 'aten::exp': None, 'aten::flip': None}
    except Exception:
        pass
    counts, _ = flop_count(ImageEncoder(model).eval(), (torch.randn(1, 3, res, res, device=dev),), supported_ops=ops)
    return sum(counts.values())


@torch.no_grad()
def efficiency_once(model, res, prec):
    enc = ImageEncoder(model)
    import gc; gc.collect()
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / 2**20
    x = torch.randn(1, 3, res, res, device=dev)
    with prec_ctx(prec):
        for _ in range(5): enc(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(50):
            t = time.perf_counter(); enc(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    vram = torch.cuda.max_memory_allocated() / 2**20
    # throughput + energy: largest batch <= 64 with batch*res^2 <= 64*384^2, >= 3 s of work
    b = 64
    while b > 1 and b * res * res > 64 * 384 * 384:
        b //= 2
    while True:
        try:
            xb = torch.randn(b, 3, res, res, device=dev)
            with prec_ctx(prec):
                enc(xb); torch.cuda.synchronize()
                n, e0, t0 = 0, energy_mj() if energy_mj else 0, time.perf_counter()
                while time.perf_counter() - t0 < 3.0:
                    enc(xb); n += 1
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                e = ((energy_mj() - e0) / 1000.0) if energy_mj else float('nan')
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); b //= 2
            if b == 0: return None
    return dict(latency_ms=statistics.median(ts) * 1000, vram_peak_mb=vram, vram_act_mb=vram - base,
                throughput_img_s=n * b / dt, energy_j_per_img=e / (n * b), batch=b)


def agg(runs):
    out = {}
    for k in runs[0]:
        v = [r[k] for r in runs]
        out[k] = {'mean': statistics.mean(v), 'std': statistics.stdev(v) if len(v) > 1 else 0.0}
    return out


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--parts', default='acc,res,prec,eff')
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()
    path = f'{OUT_DIR}/{args.model}.json'
    R = json.load(open(path)) if os.path.exists(path) else {}
    save = lambda: json.dump(R, open(path, 'w'), indent=1)

    model = load_model(args.model).to(dev).eval()
    if hasattr(model, 'raw'):
        model.raw = True
    R['name'] = model.name
    R['params_M'] = sum(p.numel() for p in model.parameters()) / 1e6
    print('Model:', model.name, f'{R["params_M"]:.0f}M params', flush=True)
    parts = args.parts.split(',')

    if 'acc' in parts and 'acc' not in R:
        t0 = time.time()
        R['acc'] = {'coco5k': retrieval(model, coco5k_items(), 224, 'fp32'),
                    'nocaps': retrieval(model, nocaps_items(), 224, 'fp32'),
                    'imagenet': imagenet(model, 224, 'fp32')}
        print('acc', json.dumps(R['acc']), f'{time.time()-t0:.0f}s', flush=True); save()

    if 'res' in parts:
        R.setdefault('imagenet_res', {})
        for res in (128, 224, 384, 512):
            if str(res) not in R['imagenet_res']:
                R['imagenet_res'][str(res)] = imagenet(model, res, 'fp32')
                print('imagenet res', res, R['imagenet_res'][str(res)], flush=True); save()

    if 'prec' in parts:
        R.setdefault('precision', {})
        if 'fp32' not in R['precision'] and 'acc' in R:
            R['precision']['fp32'] = {'coco5k': R['acc']['coco5k'], 'imagenet': R['acc']['imagenet']}
        for prec in ('bf16', 'int8', 'int4'):
            if prec in R['precision']:
                continue
            m = model
            if prec in ('int8', 'int4'):
                m = copy.deepcopy(model)
                nq = fake_quant_(m, 8 if prec == 'int8' else 4)
                print(f'{prec}: quantised {nq} weight tensors', flush=True)
            R['precision'][prec] = {'coco5k': retrieval(m, coco5k_items(), 224, 'bf16' if prec == 'bf16' else 'fp32'),
                                    'imagenet': imagenet(m, 224, 'bf16' if prec == 'bf16' else 'fp32')}
            if m is not model:
                del m; import gc; gc.collect(); torch.cuda.empty_cache()
            print('precision', prec, json.dumps(R['precision'][prec]), flush=True); save()

    if 'eff' in parts:
        R.setdefault('efficiency', {})
        for res in (64, 128, 224, 256, 512, 1024):
            for prec in ('fp32', 'bf16'):
                key = f'{res}_{prec}'
                if key in R['efficiency']:
                    continue
                runs = [efficiency_once(model, res, prec) for _ in range(args.repeats)]
                if any(r is None for r in runs):
                    R['efficiency'][key] = 'OOM'
                else:
                    R['efficiency'][key] = agg([{k: v for k, v in r.items() if k != 'batch'} for r in runs])
                    R['efficiency'][key]['batch'] = runs[0]['batch']
                    try:
                        R['efficiency'][key]['gflops'] = gflops(model, res)
                    except Exception as e:
                        R['efficiency'][key]['gflops'] = None; print('flops failed', res, e)
                print('eff', key, json.dumps(R['efficiency'][key]), flush=True); save()
    print('NOTES_DONE', flush=True)


if __name__ == '__main__':
    main()
