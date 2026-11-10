"""
Evaluate a fine-tuned CLIMP checkpoint on image-text retrieval:
  * COCO 5K (val2017, 5,000 images / ~25K captions)
  * NoCaps validation (4,500 images / 45,000 captions)

Usage: python eval_climp.py [--ckpt runs/v3_ft/climp_ft_best.pth] [--out runs/v3_ft/eval]
"""
import os, json, time, argparse
import torch
from climp_model import (ROOT, VisionMambaEncoder, TextMambaEncoder, CLIMP, build_tokenizer,
                         load_coco5k, load_nocaps_val, retrieval_eval)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{ROOT}/runs/v3_ft/climp_ft_best.pth')
    ap.add_argument('--out', default=None)
    ap.add_argument('--coco', default=f'{ROOT}/data/coco')
    ap.add_argument('--nocaps', default=f'{ROOT}/data/nocaps')
    ap.add_argument('--text_cfg', default=f'{ROOT}/ckpt/mamba-final/config.json')
    ap.add_argument('--tokenizer', default=f'{ROOT}/ckpt/mamba-final/tokenizer.json')
    ap.add_argument('--datasets', default='coco5k,nocaps')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt), 'eval')
    os.makedirs(out_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device('cuda')
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck.get('cfg', {})
    model = CLIMP(VisionMambaEncoder(None, use_checkpoint=False),
                  TextMambaEncoder(args.text_cfg, None),
                  cfg.get('EMBED_DIM', 512), vis_amp=cfg.get('VIS_AMP', 0))
    model.load_state_dict(ck['model'], strict=True)
    model.to(dev).eval()
    tok = build_tokenizer(args.tokenizer)
    print(f'Loaded {args.ckpt}  (epoch {ck.get("epoch", -1) + 1}, step {ck.get("step")})', flush=True)

    loaders = {'coco5k': lambda: load_coco5k(args.coco), 'nocaps': lambda: load_nocaps_val(args.nocaps)}
    names = {'coco5k': 'COCO 5K', 'nocaps': 'NoCaps val'}
    results = {}
    for ds in args.datasets.split(','):
        t0 = time.time()
        r = retrieval_eval(model, tok, *loaders[ds](), dev, max_len=cfg.get('MAX_SEQ_LEN', 77),
                           ks=(1, 3, 5, 10), num_workers=args.workers, mp_context='spawn')
        r['seconds'] = round(time.time() - t0, 1)
        results[ds] = r
        print(f'{names[ds]}: ' + '  '.join(f'{k}={v:.2f}' for k, v in r.items()
                                           if 'R@' in k or k == 'Rsum'), flush=True)

    json.dump(dict(ckpt=args.ckpt, epoch=ck.get('epoch', -1) + 1, results=results),
              open(f'{out_dir}/retrieval.json', 'w'), indent=2)

    # markdown table
    cols = ['R@1', 'R@3', 'R@5', 'R@10']
    lines = ['| Dataset | #img / #cap | ' + ' | '.join(f'I→T {c}' for c in cols) + ' | ' +
             ' | '.join(f'T→I {c}' for c in cols) + ' | Rsum |',
             '|' + '---|' * (3 + 2 * len(cols))]
    for ds, r in results.items():
        lines.append(f'| {names[ds]} | {r["n_images"]:,} / {r["n_captions"]:,} | ' +
                     ' | '.join(f'{r[f"I2T_{c}"]:.2f}' for c in cols) + ' | ' +
                     ' | '.join(f'{r[f"T2I_{c}"]:.2f}' for c in cols) + f' | {r["Rsum"]:.2f} |')
    md = '\n'.join(lines) + '\n\nRsum = sum of R@1/5/10 in both directions (v2 frozen COCO 5K baseline: 139.87)\n'
    open(f'{out_dir}/retrieval.md', 'w').write(md)
    print('\n' + md + f'\nSaved to {out_dir}/retrieval.json and retrieval.md')


if __name__ == '__main__':
    main()
