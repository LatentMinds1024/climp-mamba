"""Calibrate the (unspecified) hubness definition of CLIMP paper Table 6 against the released checkpoint.
Dumps NoCaps@224 embeddings per model, then evaluates several standard hubness definitions."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_bench import embed_pairs, CaptionDS, nocaps_items, ROOT
from models import load_model, square_transform

OUT = f'{ROOT}/bench/results/geometry_variants.json'


def skew(counts):
    c = counts.float() - counts.float().mean()
    return (c.pow(3).mean() / c.pow(2).mean().pow(1.5)).item()


def kocc(q, db, k, exclude_self=False):
    counts = torch.zeros(len(db), device=q.device)
    for i in range(0, len(q), 4096):
        s = q[i:i + 4096] @ db.T
        if exclude_self:
            r = torch.arange(i, min(i + 4096, len(q)), device=q.device)
            s[r - i, r] = -2
        counts += torch.bincount(s.topk(k, dim=1).indices.flatten(), minlength=len(db)).float()
    return counts


def variants(img, txt, t2i):
    img, txt, t2i = img.cuda(), txt.cuda(), t2i.cuda()
    first = torch.zeros(len(img), dtype=torch.long, device='cuda')        # first caption of each image
    seen = torch.zeros(len(img), dtype=torch.bool, device='cuda')
    for j, i in enumerate(t2i.tolist()):
        if not seen[i]: first[i] = j; seen[i] = True
    txt1 = txt[first]
    out = {}
    for k in (5, 10, 20):
        out[f'cross_all_k{k}'] = (skew(kocc(txt, img, k)), skew(kocc(img, txt, k)))
        out[f'cross_1cap_k{k}'] = (skew(kocc(txt1, img, k)), skew(kocc(img, txt1, k)))
        out[f'within_all_k{k}'] = (skew(kocc(img, img, k, True)), skew(kocc(txt, txt, k, True)))
        out[f'within_1cap_k{k}'] = (skew(kocc(img, img, k, True)), skew(kocc(txt1, txt1, k, True)))
    return out


def main():
    res = {}
    for name in ('paper', 'ours'):
        f = f'{ROOT}/bench/results/{name}/nocaps224_embeddings.pt'
        if not os.path.exists(f):
            m = load_model(name).cuda().eval()
            torch.save(embed_pairs(m, CaptionDS(nocaps_items(), square_transform(224)), 64, 12), f)
            del m; torch.cuda.empty_cache()
        img, txt, t2i = torch.load(f)
        res[name] = variants(img, txt, t2i)
    print(f'{"variant":18s} {"paper ckpt (img, txt)":>24s} {"ours (img, txt)":>20s}   target paper Table 6: (1.01, 1.15)')
    for v in res['paper']:
        p, o = res['paper'][v], res['ours'][v]
        print(f'{v:18s} {p[0]:10.2f} {p[1]:8.2f}      {o[0]:8.2f} {o[1]:8.2f}')
    json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
