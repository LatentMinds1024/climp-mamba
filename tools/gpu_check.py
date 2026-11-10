import os, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser('~/climp'))
from climp_model import *
torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device('cuda'); print(torch.cuda.get_device_name(0))
import causal_conv1d_cuda, selective_scan_cuda; print('kernels OK')
R = os.path.expanduser('~/climp')
tok = build_tokenizer(f'{R}/ckpt/mamba-final/tokenizer.json')
for gc in (1, 0):
    for vis_amp in (0, 1):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m = CLIMP(VisionMambaEncoder(f'{R}/ckpt/vssm_base_ckpt.pth', gc),
                  TextMambaEncoder(f'{R}/ckpt/mamba-final/config.json', f'{R}/ckpt/ckpt/best_model.pt'),
                  vis_amp=vis_amp).to(dev).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-6)
        B = int(os.environ.get('B', 128))
        t = tok(['a dog runs on the beach'] * B, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        ids, mask = t['input_ids'].to(dev), t['attention_mask'].to(dev)
        x = torch.randn(B, 3, 224, 224, device=dev)
        try:
            for i in range(4):
                if i == 1: torch.cuda.synchronize(); t0 = time.time()
                a, b, s = m(x, ids, mask)
                lg = s * a @ b.T; lab = torch.arange(B, device=dev)
                loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
                loss.backward(); opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            print(f'grad_ckpt={gc} vis_bf16={vis_amp} B={B}: loss={loss.item():.3f} '
                  f'{(time.time()-t0)/3:.2f}s/it peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB', flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f'grad_ckpt={gc} vis_bf16={vis_amp} B={B}: OOM', flush=True)
        del m, opt
