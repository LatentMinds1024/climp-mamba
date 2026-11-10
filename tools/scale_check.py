"""Memory/throughput of one CLIMP training step: VMamba-B (bf16 autocast, no grad-ckpt) +
Mamba-1 text towers of increasing size (random init; same shapes as state-spaces/mamba-*), AdamW."""
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser('~/climp'))
from climp_model import VisionMambaEncoder, ProjectionHead
from mamba_ssm.models.mixer_seq_simple import MixerModel
torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device('cuda')
SIZES = {'99M-inhouse': (768, 16), '130m': (768, 24), '370m': (1024, 48), '790m': (1536, 48),
         '1.4b': (2048, 48), '2.8b': (2560, 64)}
B = int(os.environ.get('B', 128))
for name in os.environ.get('SIZES', ','.join(SIZES)).split(','):
    d, n = SIZES[name]
    for L in (77, 256):
        for mode in ('full', 'text_frozen'):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                vis = VisionMambaEncoder(None, use_checkpoint=False, log=lambda *a: None).to(dev)
                txt = MixerModel(d_model=d, n_layer=n, d_intermediate=0, vocab_size=50280,
                                 ssm_cfg={'layer': 'Mamba1'}, rms_norm=True, fused_add_norm=True,
                                 residual_in_fp32=True).to(dev)
                vp, tp = ProjectionHead(1024, 512).to(dev), ProjectionHead(d, 512).to(dev)
                if mode == 'text_frozen':
                    txt.requires_grad_(False)
                params = [p for m in (vis, txt, vp, tp) for p in m.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=1e-6, fused=True)
                x = torch.randn(B, 3, 224, 224, device=dev); ids = torch.randint(0, 50000, (B, L), device=dev)
                for i in range(4):
                    if i == 1: torch.cuda.synchronize(); t0 = time.time()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        a = F.normalize(vp(vis(x)).float(), dim=-1)
                        with torch.set_grad_enabled(mode == 'full'):
                            h = txt(ids).mean(1)
                        b = F.normalize(tp(h).float(), dim=-1)
                    lg = 14 * a @ b.T; lab = torch.arange(B, device=dev)
                    loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
                    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); dt = (time.time() - t0) / 3
                ntxt = sum(p.numel() for p in txt.parameters()) / 1e6
                print(f'{name:12s} ({ntxt:6.0f}M) L={L:3d} {mode:11s} B={B}: {dt:.2f}s/it '
                      f'{B/dt:6.0f} img/s/GPU  peak={torch.cuda.max_memory_allocated()/2**30:5.1f}GB', flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f'{name:12s} L={L:3d} {mode:11s} B={B}: OOM', flush=True)
            for v in ('vis', 'txt', 'vp', 'tp', 'opt', 'x', 'ids', 'a', 'b', 'h', 'lg', 'loss'):
                globals().pop(v, None)
