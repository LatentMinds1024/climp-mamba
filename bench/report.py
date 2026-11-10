"""Collect benchmark results -> side-by-side comparison (markdown + JSON) against the CLIMP paper."""
import os, sys, json, glob, re, statistics

ROOT = os.path.expanduser('~/climp')
RES_DIR = f'{ROOT}/bench/results'
TXT = open(f'{ROOT}/paper/text.txt').read()
MODELS = [('paper', 'Paper ckpt (our eval)'), ('ours', 'Ours')]

# ───────────── paper numbers, parsed from the arXiv HTML text (Tables 9, 10) ─────────────
T9_COLS = ['ViT+LLaMA', 'CLIMP-Mamba2', 'CLIMP-Mamba1', 'RoPE-ViT+LLaMA', 'FlexViT+LLaMA', 'NaFlex+LLaMA']
CLS = ['cars', 'country211', 'fer2013', 'fgvc_aircraft', 'imagenet-a', 'imagenet-o', 'imagenet-r', 'imagenet1k',
       'imagenet_sketch', 'imagenetv2', 'objectnet', 'sun397', 'voc2007', 'cifar100', 'cifar10',
       'clevr_closest_object_distance', 'clevr_count_all', 'diabetic_retinopathy', 'dmlab',
       'dsprites_label_orientation', 'dsprites_label_x_position', 'dsprites_label_y_position',
       'smallnorb_label_azimuth', 'smallnorb_label_elevation', 'svhn']
RET = ['flickr30k', 'flickr8k', 'mscoco_captions']


def parse_rows(block, names, ncols):
    out = {}
    for n in names:
        m = re.search(r'(?<![\w-])' + re.escape(n) + r'((?:\s+[\d.]+){%d})' % ncols, block)
        if m:
            out[n] = [float(v) for v in m.group(1).split()]
    return out


t9 = TXT[TXT.find('A.2 Main Results'):TXT.find('Table 9:')]
t10 = TXT[TXT.find('Table 9:'):TXT.find('Table 10:')]
PAPER_T9 = parse_rows(t9, CLS, 12)       # 6 models x (acc1, acc5)
PAPER_T10 = parse_rows(t10, RET, 12)     # 6 models x (T-R@5, I-R@5)
P = T9_COLS.index('CLIMP-Mamba1')

PAPER = {  # other tables, CLIMP (VMamba) Mamba-1 row + best transformer baseline
    'T1': {'CLIMP-Mamba1': (29.6, 58.5, 65.5, 77.0), 'RoPE-ViT-B/16': (27.3, 59.0, 63.4, 72.9)},
    'T3': {'CLIMP-Mamba1': {224: (29.6, 58.5, 65.5, 77.0), 320: (28.4, 57.5, 66.0, 76.9), 384: (27.3, 56.3, 65.0, 75.2)}},
    # Table 11 as printed (columns labelled I->T, T->I); see note about the label swap in the report
    'T11': {'nocaps': {224: (70.5, 81.9), 512: (66.9, 72.7), 896: (57.1, 53.5)},
            'xm3600': {224: (65.1, 69.1), 512: (62.6, 60.8), 896: (55.9, 45.4)}},
    'T11_rope': {'nocaps': {224: (67.6, 78.9), 512: (65.1, 75.9), 896: (38.6, 34.9)},
                 'xm3600': {224: (64.4, 68.2), 512: (62.3, 64.1), 896: (36.8, 26.9)}},
    'T13': {'image_retrieval': (37.3, 67.1, 77.8), 'text_retrieval': (23.8, 50.4, 62.7)},
    'T13_naflex': {'image_retrieval': (29.1, 55.3, 67.5), 'text_retrieval': (26.1, 53.4, 65.1)},
    'T12': {'image_retrieval': (67.0, 89.4, 94.7), 'text_retrieval': (81.3, 95.9, 98.4)},
    'T6': {'CLIMP-Mamba1': (0.982, -3.18, -3.54, 1.01, 1.15), 'RoPE-ViT': (1.052, -3.16, -3.52, 1.04, 1.25)},
    'README_nocaps224': {'image_retrieval': (38.2, 70.5, 81.8), 'text_retrieval': (51.8, 81.9, 90.9)},
}


def load(model):
    out = {}
    for f in glob.glob(f'{RES_DIR}/{model}/*.json'):
        r = json.load(open(f)); out[(r['task'], r['dataset'].split('/')[-1], r['res'])] = r
    return out


R = {m: load(m) for m, _ in MODELS}
f1 = lambda v: '—' if v is None else f'{v:.1f}'
f3 = lambda v: '—' if v is None else f'{v:.3f}'


def get(m, task, ds, res, key, scale=100):
    r = R[m].get((task, ds, res))
    return None if r is None or key not in r else r[key] * scale


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def complete(xs):
    return all(x is not None for x in xs)


L = []
w = L.append
w('# CLIMP: paper vs. released checkpoint (our eval) vs. ours\n')
w('Protocol for all evaluated columns: authors\' CLIMP repo (square bicubic resize, ImageNet norm, no text truncation, fp16 autocast), CLIP-Benchmark templates/classnames/metrics. "Image retrieval" = text→image (T→I); "text retrieval" = image→text (I→T).\n')

# ── sanity check vs README ──
w('## 0. Reproduction check — NoCaps val @224 vs authors\' README\n')
w('| | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 |\n|---|---|---|---|---|---|---|')
rd = PAPER['README_nocaps224']
w('| Authors\' README | ' + ' | '.join(f1(v) for v in rd['image_retrieval'] + rd['text_retrieval']) + ' |')
for m, lab in MODELS:
    w(f'| {lab} | ' + ' | '.join(f1(get(m, 'hires', 'nocaps', 224, f'image_retrieval_recall@{k}')) for k in (1, 5, 10)) +
      ' | ' + ' | '.join(f1(get(m, 'hires', 'nocaps', 224, f'text_retrieval_recall@{k}')) for k in (1, 5, 10)) + ' |')
w('')

# ── Table 1 ──
w('## 1. CLIP-Benchmark summary (Table 1) @224\n')
w('Acc = mean over the 25 classification datasets of paper Table 9; IR@5/TR@5 = mean over flickr30k, flickr8k, mscoco (Table 10).\n')
w('| Model | Acc@1 | Acc@5 | IR@5 (T→I) | TR@5 (I→T) |\n|---|---|---|---|---|')
pa1 = mean([PAPER_T9[d][2 * P] for d in CLS if d in PAPER_T9]); pa5 = mean([PAPER_T9[d][2 * P + 1] for d in CLS if d in PAPER_T9])
ptr = mean([PAPER_T10[d][2 * P] for d in RET]); pir = mean([PAPER_T10[d][2 * P + 1] for d in RET])
w(f'| Paper, CLIMP-Mamba1 (Table 1 as printed) | 29.6 | 58.5 | 65.5 | 77.0 |')
w(f'| Paper, CLIMP-Mamba1 (recomputed from Tables 9/10, same 25+3 datasets) | {f1(pa1)} | {f1(pa5)} | {f1(pir)} | {f1(ptr)} |')
w(f'| Paper, best transformer (RoPE-ViT-B/16 + LLaMA-1B) | 27.3 | 59.0 | 63.4 | 72.9 |')
for m, lab in MODELS:
    a1 = [get(m, 'cb', d, 224, 'acc1') for d in CLS]; a5 = [get(m, 'cb', d, 224, 'acc5') for d in CLS]
    ir = [get(m, 'cb', d, 224, 'image_retrieval_recall@5') for d in RET]; tr = [get(m, 'cb', d, 224, 'text_retrieval_recall@5') for d in RET]
    tag = '' if complete(a1) else f' (partial: {sum(x is not None for x in a1)}/25)'
    w(f'| {lab}{tag} | {f1(mean(a1))} | {f1(mean(a5))} | {f1(mean(ir))} | {f1(mean(tr))} |')
w('')

# ── Table 2 ──
w('## 2. OOD robustness (Table 2) @224, Acc@1 / Acc@5\n')
IV = ['imagenetv2', 'imagenet-r', 'imagenet-a', 'imagenet-o', 'imagenet_sketch']
w('| Model | IN-1k | IN-V2 | IN-R | IN-A | IN-O | Sketch | Avg (5 variants) |\n|---|---|---|---|---|---|---|---|')
row = lambda a, b: f'{f1(a)}/{f1(b)}'
w('| Paper CLIMP-Mamba1 (Table 9 cols) | ' + ' | '.join(row(*PAPER_T9[d][2 * P:2 * P + 2]) for d in ['imagenet1k'] + IV) +
  f' | {row(mean([PAPER_T9[d][2*P] for d in IV]), mean([PAPER_T9[d][2*P+1] for d in IV]))} |')
for m, lab in MODELS:
    vals = [(get(m, 'cb', d, 224, 'acc1'), get(m, 'cb', d, 224, 'acc5')) for d in ['imagenet1k'] + IV]
    w(f'| {lab} | ' + ' | '.join(row(*v) for v in vals) +
      f' | {row(mean([v[0] for v in vals[1:]]), mean([v[1] for v in vals[1:]]))} |')
w('\nNote: the paper\'s Table 2 and Table 9 disagree on which CLIMP variant has IN-O 48.1 vs 49.8; Table 9 values are shown.\n')

# ── Table 9 per dataset ──
w('## 3. Per-dataset zero-shot classification (Table 9) @224, Acc@1\n')
w('| Dataset | Paper CLIMP-M1 | Paper best transformer | ' + ' | '.join(l for _, l in MODELS) + ' |\n|---|---|---|' + '---|' * len(MODELS))
for d in CLS:
    pv = PAPER_T9.get(d)
    best_tf = max(pv[0], pv[6], pv[8], pv[10]) if pv else None
    w(f'| {d} | {f1(pv[2*P]) if pv else "—"} | {f1(best_tf)} | ' +
      ' | '.join(f1(get(m, 'cb', d, 224, 'acc1')) for m, _ in MODELS) + ' |')
w('')

# ── Table 10 ──
w('## 4. Zero-shot retrieval (Table 10) @224 — R@1 / R@5 / R@10\n')
w('| Dataset | Direction | Paper CLIMP-M1 R@5 | ' + ' | '.join(l for _, l in MODELS) + ' |\n|---|---|---|' + '---|' * len(MODELS))
for d in RET:
    for key, lab, pidx in (('text_retrieval', 'I→T', 2 * P), ('image_retrieval', 'T→I', 2 * P + 1)):
        w(f'| {d} | {lab} | {f1(PAPER_T10[d][pidx])} | ' + ' | '.join(
            '/'.join(f1(get(m, 'cb', d, 224, f'{key}_recall@{k}')) for k in (1, 5, 10)) for m, _ in MODELS) + ' |')
w('')

# ── Table 3 ──
w('## 5. Resolution scaling on CLIP-Benchmark (Table 3) — Acc@1/Acc@5 and IR@5/TR@5\n')
w('| Model | 224 | 320 | 384 |\n|---|---|---|---|')
w('| Paper CLIMP-Mamba1 | ' + ' | '.join(f'{v[0]}/{v[1]} · {v[2]}/{v[3]}' for v in PAPER['T3']['CLIMP-Mamba1'].values()) + ' |')
for m, lab in MODELS:
    cells = []
    for res in (224, 320, 384):
        a1 = mean([get(m, 'cb', d, res, 'acc1') for d in CLS]); a5 = mean([get(m, 'cb', d, res, 'acc5') for d in CLS])
        ir = mean([get(m, 'cb', d, res, 'image_retrieval_recall@5') for d in RET]); tr = mean([get(m, 'cb', d, res, 'text_retrieval_recall@5') for d in RET])
        cells.append(f'{f1(a1)}/{f1(a5)} · {f1(ir)}/{f1(tr)}')
    w(f'| {lab} | ' + ' | '.join(cells) + ' |')
w('')

# ── Table 4/11 ──
w('## 6. High-resolution retrieval, NoCaps & Crossmodal-3600 (Tables 4, 11) — R@5\n')
w('Paper Table 11 is printed with columns "I→T, T→I", but the authors\' README for the same model gives NoCaps@224 image-retrieval (T→I) R@5 = 70.5 and text-retrieval (I→T) R@5 = 81.9, i.e. the printed labels are swapped. Below the paper values are placed under the corrected direction.\n')
w('| Dataset | Res | Paper CLIMP-M1 T→I / I→T | Paper RoPE-ViT T→I / I→T | ' + ' | '.join(f'{l} T→I / I→T' for _, l in MODELS) + ' |\n|---|---|---|---|' + '---|' * len(MODELS))
for d in ('nocaps', 'xm3600'):
    for res in (224, 512, 896):
        pv = PAPER['T11'][d][res]; rv = PAPER['T11_rope'][d][res]
        w(f'| {d} | {res} | {pv[0]} / {pv[1]} | {rv[0]} / {rv[1]} | ' + ' | '.join(
            f'{f1(get(m, "hires", d, res, "image_retrieval_recall@5"))} / {f1(get(m, "hires", d, res, "text_retrieval_recall@5"))}'
            for m, _ in MODELS) + ' |')
w('')

# ── Table 5/12/13 ──
w('## 7. Dense / long-caption retrieval (Tables 5, 12, 13) — R@1/R@5/R@10\n')
w('| Benchmark | Res | Direction | Paper CLIMP-M1 | Paper best transformer | ' + ' | '.join(l for _, l in MODELS) + ' |\n|---|---|---|---|---|' + '---|' * len(MODELS))
for res in (896, 224):
    for key, lab in (('image_retrieval', 'T→I'), ('text_retrieval', 'I→T')):
        pv = '/'.join(f1(v) for v in PAPER['T13'][key]) if res == 896 else '—'
        bv = '/'.join(f1(v) for v in PAPER['T13_naflex'][key]) + ' (NaFlex)' if res == 896 else '—'
        w(f'| DOCCI test (5k) | {res} | {lab} | {pv} | {bv} | ' + ' | '.join(
            '/'.join(f1(get(m, 'docci', 'docci', res, f'{key}_recall@{k}')) for k in (1, 5, 10)) for m, _ in MODELS) + ' |')
for key, lab in (('image_retrieval', 'T→I'), ('text_retrieval', 'I→T')):
    w(f'| Urban1k (extra) | 224 | {lab} | — | — | ' + ' | '.join(
        '/'.join(f1(get(m, 'docci', 'urban1k', 224, f'{key}_recall@{k}')) for k in (1, 5, 10)) for m, _ in MODELS) + ' |')
w(f'| Flickr8k-Rephrased | 224 | T→I / I→T | {"/".join(f1(v) for v in PAPER["T12"]["image_retrieval"])} · {"/".join(f1(v) for v in PAPER["T12"]["text_retrieval"])} | — | not released | not released |')
w('')

# ── Table 6 ──
w('## 8. Embedding geometry on NoCaps (Table 6) — lower is better\n')
w('Alignment = E‖f(x)−g(y)‖² over matched pairs (α=2); uniformity = log E exp(−2‖u−v‖²) (t=2); hubness = skewness of the k-occurrence distribution, k=10, cross-modal. The paper does not state k/t; these are the standard defaults.\n')
w('| Model | Alignment | Unif. image | Unif. text | Hub. image | Hub. text |\n|---|---|---|---|---|---|')
for k, v in PAPER['T6'].items():
    w(f'| Paper {k} | ' + ' | '.join(f3(x) if i == 0 else f'{x:.2f}' for i, x in enumerate(v)) + ' |')
for m, lab in MODELS:
    r = R[m].get(('geometry', 'nocaps', 224))
    w(f'| {lab} | ' + (' | '.join(f3(r[k]) for k in ('alignment', 'uniformity_image', 'uniformity_text', 'hubness_image', 'hubness_text')) if r else '— | — | — | — | —') + ' |')
w('')

# ── efficiency ──
ef = f'{RES_DIR}/efficiency.json'
w('## 9. Efficiency (Fig. 3 + latency / throughput / energy)\n')
if os.path.exists(ef):
    E = json.load(open(ef))
    w(f'GPU: {E["gpu"]}. Vision towers, batch 1 (memory = resolution-specific peak above resident weights).\n')
    w('| Tower | Res | Params (M) | GFLOPs | Mem fp32 (MB) | Lat fp32 (ms) | Lat bf16 (ms) | Thr bf16 (img/s) | Energy bf16 (J/img) |\n|---|---|---|---|---|---|---|---|---|')
    for r in E['vision']:
        g = lambda k: r.get(k)
        lat = lambda k: f'{g(k):.1f}' if isinstance(g(k), float) else str(g(k))
        thr = r.get('throughput_bf16', {})
        w(f'| {r["tower"]} | {r["res"]} | {r["params_m"]:.1f} | {r["gflops"]:.1f} | '
          f'{lat("peak_mem_mb_fp32")} | {lat("latency_ms_fp32")} | {lat("latency_ms_bf16")} | '
          f'{thr.get("items_per_s", 0):.0f} | {lat("energy_j_per_item_bf16")} |')
    w('\nText towers (batch 1):\n')
    w('| Tower | Tokens | Params (M) | ~GFLOPs | Lat bf16 (ms) | Mem bf16 (MB) | Thr bf16 (captions/s) |\n|---|---|---|---|---|---|---|')
    for r in E['text']:
        w(f'| {r["tower"]} | {r["tokens"]} | {r["params_m"]:.0f} | {r["gflops_est"]:.1f} | {r["latency_ms_bf16"]:.1f} | '
          f'{r["peak_mem_mb_bf16"]:.0f} | {r["throughput_bf16"]["items_per_s"]:.0f} |')
    w('\nPaper (Fig. 3 text): at 896×896 CLIMP 10.0 MB vs ViT 50.4 MB resolution-specific memory; 259.7 vs 457.9 GFLOPs.\n')
else:
    w('_pending_\n')

q = f'{RES_DIR}/qualitative.json'
w('## 10. Qualitative similarity (Fig. 2, 5) — same NoCaps images and captions as the paper\n')
if os.path.exists(q):
    Q = json.load(open(q))
    w('| Caption | Paper CLIMP score | ' + ' | '.join(l for _, l in MODELS) + ' |\n|---|---|' + '---|' * len(MODELS))
    for i, ex in enumerate(Q['paper']):
        w(f'| {ex["caption"]} | {ex["paper_score"]:.3f} | ' + ' | '.join(f'{Q[m][i]["score"]:.3f}' for m, _ in MODELS) + ' |')
    w('\nMaps: bench/results/qualitative.png\n')
else:
    w('_pending_\n')

md = '\n'.join(L)
open(f'{RES_DIR}/comparison.md', 'w').write(md)
print(md)
