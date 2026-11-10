"""
Efficiency benchmark (CLIMP paper Fig. 3 + user's quantitative list).

Vision towers vs resolution: VMamba-B (identical architecture in CLIMP-paper and CLIMP-ours)
vs a ViT-B/16 reference (naive attention as in timm, and fused SDPA attention).
Text towers vs caption length: in-house Mamba-1 99M (ours) vs Mamba-1 1.4B (paper).

Metrics: params, GFLOPs (fvcore; VMamba selective-scan counted with the official VMamba hook),
resolution-specific peak memory (peak during forward minus resident weights, batch 1), latency
(batch 1, median of 30), throughput (img/s at the largest power-of-2 batch <= 64 that fits),
energy per image (NVML total-energy counter, J).
"""
import os, sys, json, time, math, statistics
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ROOT, VMAMBA_BASE_CONFIG
from vmamba import VSSM
from fvcore.nn import flop_count

dev = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True
RES = [int(r) for r in os.environ.get('RES', '64,128,224,256,384,512,768,896,1024').split(',')]
OUT = f'{ROOT}/bench/results/efficiency.json'

try:
    import pynvml
    pynvml.nvmlInit()
    H = pynvml.nvmlDeviceGetHandleByIndex(0)   # SLURM exposes the allocated GPU only (index 0 here)
    def energy_mj(): return pynvml.nvmlDeviceGetTotalEnergyConsumption(H)
except Exception as e:
    print('NVML energy unavailable:', e); energy_mj = None


def vmamba_b():
    m = VSSM(**VMAMBA_BASE_CONFIG); delattr(m.classifier, 'head'); return m


def vit_b16(res, sdpa):
    from timm.models.vision_transformer import VisionTransformer, Attention
    m = VisionTransformer(img_size=res, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0)
    if sdpa:
        def fwd(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
            return self.proj(x.transpose(1, 2).reshape(B, N, C))
        for mod in m.modules():
            if isinstance(mod, Attention):
                mod.forward = fwd.__get__(mod)
    return m


def gflops(model, x, vmamba):
    if vmamba:
        return model.flops(tuple(x.shape[1:]), verbose=False) / 1e9
    counts, _ = flop_count(model, (x,))
    return sum(counts.values())


@torch.no_grad()
def measure(model, make_input, amp_dtype=None, batch=1, reps=30, min_seconds=0.0):
    """Latency (median), resolution-specific peak memory, and energy per item.
    Runs at least `reps` iterations and at least `min_seconds` (NVML energy counter is coarse)."""
    model.to(dev).eval()
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    x = make_input(batch)
    ctx = torch.autocast('cuda', dtype=amp_dtype) if amp_dtype else torch.autocast('cuda', enabled=False)
    with ctx:
        for _ in range(5): model(x)
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20
        ts, n = [], 0
        e0 = energy_mj() if energy_mj else None
        t_all = time.time()
        while n < reps or time.time() - t_all < min_seconds:
            t = time.time(); model(x); torch.cuda.synchronize(); ts.append(time.time() - t); n += 1
        t_all = time.time() - t_all
        e = ((energy_mj() - e0) / 1000.0) if energy_mj else float('nan')
    return dict(latency_ms=statistics.median(ts) * 1000, peak_mem_mb=peak,
                energy_j_per_item=e / (n * batch), wall_s=t_all, iters=n)


MAX_PIXELS = 64 * 384 * 384      # cap batch*H*W: selective-scan kernels hit int32 indexing limits beyond this


def throughput(model, make_input, amp_dtype, pixels=None):
    """Items/s and energy/item at the largest batch (<=64, <=MAX_PIXELS) that fits; >=3 s of work."""
    for b in (64, 32, 16, 8, 4, 2, 1):
        if pixels and b * pixels > MAX_PIXELS and b > 1:
            continue
        try:
            r = measure(model, make_input, amp_dtype, batch=b, reps=10, min_seconds=3.0)
            return dict(batch=b, items_per_s=b / (r['latency_ms'] / 1000), energy_j_per_item=r['energy_j_per_item'])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return dict(batch=0, items_per_s=0.0, energy_j_per_item=float('nan'))


def save(results):
    json.dump(results, open(OUT, 'w'), indent=1)


def main():
    results = {'vision': [], 'text': [], 'gpu': torch.cuda.get_device_name(0)}
    # ── vision towers ──
    for res in RES:
        towers = [('VMamba-B (CLIMP)', lambda: vmamba_b(), True)]
        if res % 16 == 0:
            towers += [('ViT-B/16 naive-attn', lambda: vit_b16(res, False), False),
                       ('ViT-B/16 SDPA', lambda: vit_b16(res, True), False)]
        for name, build, is_vm in towers:
            m = build().to(dev).eval()
            params = sum(p.numel() for p in m.parameters()) / 1e6
            mk = lambda b, r=res: torch.randn(b, 3, r, r, device=dev)
            row = dict(tower=name, res=res, params_m=params)
            try:
                row['gflops'] = gflops(m, mk(1), is_vm)
            except Exception as e:
                row['gflops'] = float('nan'); print('flops fail', name, res, e)
            for prec, dt in (('fp32', None), ('bf16', torch.bfloat16)):
                try:
                    r = measure(m, mk, dt)
                    row.update({f'{k}_{prec}': v for k, v in r.items()})
                    row.update({f'throughput_{prec}': throughput(m, mk, dt, res * res)})
                except torch.cuda.OutOfMemoryError:
                    row[f'latency_ms_{prec}'] = 'OOM'; torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            results['vision'].append(row); save(results)
            del m; torch.cuda.empty_cache()

    # ── text towers ──
    from mamba_ssm.models.mixer_seq_simple import MixerModel
    TXT = {'in-house Mamba-1 99M (ours)': (768, 16), 'Mamba-1 1.4B (paper)': (2048, 48)}
    for name, (d, n) in TXT.items():
        m = MixerModel(d_model=d, n_layer=n, d_intermediate=0, vocab_size=50280, ssm_cfg={'layer': 'Mamba1'},
                       rms_norm=True, fused_add_norm=True, residual_in_fp32=True).to(dev).eval()
        params = sum(p.numel() for p in m.parameters()) / 1e6
        nonemb = params - 50280 * d / 1e6
        for L in (16, 77, 142, 256, 512):
            mk = lambda b, L=L: torch.randint(0, 50000, (b, L), device=dev)
            row = dict(tower=name, tokens=L, params_m=params,
                       gflops_est=2 * nonemb * 1e6 * L / 1e9)   # 2*N*L for the dense projections
            for prec, dt in (('fp32', None), ('bf16', torch.bfloat16)):
                r = measure(m, mk, dt)
                row.update({f'{k}_{prec}': v for k, v in r.items()})
                row.update({f'throughput_{prec}': throughput(m, mk, dt)})
            print(json.dumps(row), flush=True)
            results['text'].append(row); save(results)
        del m; torch.cuda.empty_cache()
    json.dump(results, open(OUT, 'w'), indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
