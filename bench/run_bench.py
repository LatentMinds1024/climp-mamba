"""
CLIMP paper benchmark suite (arXiv 2601.06891), run identically for every model.

Tasks (paper table in brackets):
  cb       CLIP-Benchmark zero-shot classification on the 25 Table-9 datasets (Acc@1/Acc@5)
           + zero-shot retrieval flickr30k / flickr8k / mscoco_captions (R@1/5/10)   [T1, T2, T3, T9, T10]
  hires    NoCaps-val + Crossmodal-3600 (EN) retrieval, R@1/5/10                       [T4, T11]
  docci    DOCCI test (5k) long-caption retrieval, R@1/5/10                            [T5, T13]
  geometry NoCaps-val embedding geometry: alignment, uniformity, hubness               [T6]

Protocol (authors' CLIMP repo): square Resize((r,r)) bicubic, ImageNet mean/std, no text
truncation, fp16 autocast (CLIP-Benchmark default), CLIP-Benchmark templates/classnames/metrics.

Usage: python bench/run_bench.py --model ours|paper --task cb --res 224 [--datasets a,b]
"""
import os, sys, json, time, argparse
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load_model, square_transform, ROOT
from clip_benchmark.datasets.builder import build_dataset, image_captions_collate_fn
from clip_benchmark.metrics import zeroshot_classification as zsc, zeroshot_retrieval as zsr

CB_ROOT = f'{ROOT}/data/clip_benchmark'
CLS_DATASETS = ['cars', 'country211', 'fer2013', 'fgvc_aircraft', 'imagenet-a', 'imagenet-o', 'imagenet-r',
                'imagenet1k', 'imagenet_sketch', 'imagenetv2', 'objectnet', 'sun397', 'voc2007',
                'vtab/cifar100', 'vtab/cifar10', 'vtab/clevr_closest_object_distance', 'vtab/clevr_count_all',
                'vtab/diabetic_retinopathy', 'vtab/dmlab', 'vtab/dsprites_label_orientation',
                'vtab/dsprites_label_x_position', 'vtab/dsprites_label_y_position',
                'vtab/smallnorb_label_azimuth', 'vtab/smallnorb_label_elevation', 'vtab/svhn']
RET_DATASETS = ['flickr30k', 'flickr8k', 'mscoco_captions']
KS = [1, 5, 10]


def wds_dir(name):
    return f'{CB_ROOT}/wds_{name.replace("/", "-")}'


# ───────────────────────────── local retrieval datasets ─────────────────────────────
class CaptionDS(Dataset):
    def __init__(self, items, transform):   # items: list of (image_path, [captions])
        self.items, self.transform = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, caps = self.items[i]
        return self.transform(Image.open(p).convert('RGB')), caps


def nocaps_items():
    d = json.load(open(f'{ROOT}/data/nocaps/nocaps_val.json'))
    return [(f'{ROOT}/data/nocaps/val/{x["file_name"]}', x['captions']) for x in d]


def xm3600_items():
    out = []
    for line in open(f'{ROOT}/data/xm3600/captions_en.jsonl'):
        r = json.loads(line)
        caps = r['en']['caption'] if isinstance(r['en'], dict) else r['en']
        out.append((f'{ROOT}/data/xm3600/images/{r["image/key"]}.jpg', list(caps)))
    return out


def docci_items():
    out = []
    for line in open(f'{ROOT}/data/docci/docci_descriptions.jsonlines'):
        r = json.loads(line)
        if r['split'] == 'test':
            out.append((f'{ROOT}/data/docci/images/{r["image_file"]}', [r['description']]))
    return out


def urban1k_items():   # Long-CLIP long-caption benchmark (extra; not in the CLIMP paper)
    d = f'{ROOT}/data/urban1k/Urban1k'
    return [(f'{d}/image/{i}.jpg', [open(f'{d}/caption/{i}.txt').read().strip()]) for i in range(1, 1001)]


LOCAL = {'nocaps': nocaps_items, 'xm3600': xm3600_items, 'docci': docci_items, 'urban1k': urban1k_items}


# ───────────────────────────── runners ─────────────────────────────
def loader(ds, bs, workers, collate=None):
    return DataLoader(ds, batch_size=bs, num_workers=workers, pin_memory=True, collate_fn=collate)


def wds_workers(name, workers):
    """WebDataset splits shards across workers: never use more workers than shards."""
    return max(1, min(workers, int(open(f'{wds_dir(name)}/test/nshards.txt').read())))


def run_cls(model, name, res, bs, workers):
    ds = build_dataset(f'wds/{name}', root=wds_dir(name), transform=square_transform(res), split='test')
    m = zsc.evaluate(model, loader(ds, bs, wds_workers(name, workers)), model.tokenizer, ds.classes, ds.templates,
                     'cuda', amp=True)
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}


def run_ret(model, ds, bs, workers):
    m = zsr.evaluate(model, loader(ds, bs, workers, image_captions_collate_fn), model.tokenizer,
                     'cuda', amp=True, recall_k_list=KS)
    return {k: float(v) for k, v in m.items()}


@torch.no_grad()
def embed_pairs(model, ds, bs, workers):
    """Returns image embs (N,D), text embs (M,D), text->image index (M,) -- all L2-normalised."""
    I, T, idx = [], [], []
    for imgs, caps in loader(ds, bs, workers, image_captions_collate_fn):
        with torch.cuda.amp.autocast():
            I.append(F.normalize(model.encode_image(imgs.cuda()).float(), dim=-1).cpu())
            flat = [c for cs in caps for c in cs]
            T.append(F.normalize(model.encode_text(model.tokenizer(flat).cuda()).float(), dim=-1).cpu())
        base = sum(len(x) for x in I[:-1])
        idx += [base + i for i, cs in enumerate(caps) for _ in cs]
    return torch.cat(I), torch.cat(T), torch.tensor(idx)


def geometry(img, txt, t2i, k=10, t=2.0):
    """Wang & Isola (2020) alignment (alpha=2) and uniformity (t=2); hubness = skewness of the
    k-occurrence distribution N_k (Radovanovic et al. 2010), k=10, measured cross-modally:
    image hubness = how often each image appears in the top-k of text queries (and vice versa)."""
    dev = 'cuda'
    img, txt, t2i = img.to(dev), txt.to(dev), t2i.to(dev)
    align = (img[t2i] - txt).pow(2).sum(-1).mean().item()

    def uniformity(x, chunk=4096):
        n, tot = len(x), 0.0
        for i in range(0, n, chunk):
            d = torch.cdist(x[i:i + chunk], x).pow(2)
            e = torch.exp(-t * d)
            r = torch.arange(i, min(i + chunk, n), device=dev)
            e[r - i, r] = 0                                 # exclude i == j
            tot += e.sum().item()
        return float(torch.log(torch.tensor(tot / (n * (n - 1)))))

    def hubness(q, db, chunk=4096):
        counts = torch.zeros(len(db), device=dev)
        for i in range(0, len(q), chunk):
            nn_ = (q[i:i + chunk] @ db.T).topk(k, dim=1).indices.flatten()
            counts += torch.bincount(nn_, minlength=len(db)).float()
        c = counts - counts.mean()
        return (c.pow(3).mean() / c.pow(2).mean().pow(1.5)).item()

    return dict(alignment=align, uniformity_image=uniformity(img), uniformity_text=uniformity(txt),
                hubness_image=hubness(txt, img), hubness_text=hubness(img, txt),
                n_images=len(img), n_texts=len(txt), k=k, t=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--task', required=True, choices=['cb', 'hires', 'docci', 'geometry'])
    ap.add_argument('--res', type=int, nargs='+', default=[224])
    ap.add_argument('--datasets', default=None)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--tag', default=None, help='results subdir name (default: model)')
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = f'{ROOT}/bench/results/{args.tag or args.model.replace(":", "_").replace("/", "_")}'
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(args.model).cuda().eval()
    print('Model:', model.name, flush=True)

    jobs = []
    for res in args.res:
        if args.task == 'cb':
            names = args.datasets.split(',') if args.datasets else CLS_DATASETS + RET_DATASETS
            jobs += [('cb', n, res) for n in names]
        elif args.task == 'hires':
            jobs += [('hires', n, res) for n in (args.datasets or 'nocaps,xm3600').split(',')]
        elif args.task == 'docci':
            jobs += [('docci', n, res) for n in (args.datasets or 'docci').split(',')]
        else:
            jobs += [('geometry', 'nocaps', res)]

    for task, name, res in jobs:
        path = f'{out_dir}/{task}__{name.replace("/", "-")}__{res}.json'
        if os.path.exists(path):
            print('skip (done)', path, flush=True); continue
        t0 = time.time()
        bs = max(8, args.bs * 224 * 224 // (res * res)) if res > 384 else args.bs
        if task == 'cb' and name in RET_DATASETS:
            ds = build_dataset(f'wds/{name}', root=wds_dir(name), transform=square_transform(res), split='test',
                               task='zeroshot_retrieval')
            m = run_ret(model, ds, bs, wds_workers(name, args.workers))
        elif task == 'cb':
            m = run_cls(model, name, res, bs, args.workers)
        elif task in ('hires', 'docci'):
            m = run_ret(model, CaptionDS(LOCAL[name](), square_transform(res)), bs, args.workers)
        else:
            img, txt, t2i = embed_pairs(model, CaptionDS(LOCAL[name](), square_transform(res)), bs, args.workers)
            m = geometry(img, txt, t2i)
        m.update(model=model.name, task=task, dataset=name, res=res, seconds=round(time.time() - t0, 1))
        json.dump(m, open(path, 'w'), indent=1)
        print(f'{task:8s} {name:38s} {res:4d}  ' +
              '  '.join(f'{k}={v:.4f}' for k, v in m.items() if isinstance(v, float) and k != 'seconds') +
              f'  ({m["seconds"]}s)', flush=True)


if __name__ == '__main__':
    main()
