"""
Qualitative image-text similarity maps (CLIMP paper Fig. 2 / Fig. 5) on the exact NoCaps images
the paper used. Global score = cosine(image emb, text emb), as printed in the paper's figure.
Map = cosine between the caption embedding and the projected per-location VMamba features
(final stage, after the classifier norm, before average pooling), upsampled to the image.
"""
import os, sys, json
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load_model, square_transform, ROOT

EXAMPLES = [  # (NoCaps val file, caption, paper's CLIMP score)
    ('c9c7f104cc4e0d0e.jpg', 'A large porch with a wooden fence and no roof.', 0.545),
    ('dd5669c3169e107a.jpg', 'A woman with red hair hugging a man.', 0.628),
    ('e19707cbf772cd20.jpg', 'A hand holding a glass that has some wine in it.', 0.646),
]
RES = int(os.environ.get('QUAL_RES', 448))


@torch.no_grad()
def feature_map(model, x):
    vb = model.vision_backbone
    h = vb.patch_embed(x)
    for layer in vb.layers:
        h = layer(h)
    h = vb.classifier.norm(h)
    if h.shape[-1] != 1024:          # channel-first (ln2d) -> (B,H,W,C)
        h = h.permute(0, 2, 3, 1)
    return h


@torch.no_grad()
def run(model_name):
    m = load_model(model_name).cuda().eval()
    tf = square_transform(RES)
    out = []
    for fn, cap, paper_score in EXAMPLES:
        img = Image.open(f'{ROOT}/data/nocaps/val/{fn}').convert('RGB')
        x = tf(img)[None].cuda()
        t = F.normalize(m.encode_text(m.tokenizer([cap]).cuda()).float(), dim=-1)
        x224 = square_transform(224)(img)[None].cuda()
        score = (F.normalize(m.encode_image(x224).float(), dim=-1) @ t.T).item()
        fmap = feature_map(m, x)                                        # (1,h,w,C)
        e = F.normalize(m.project_patch_features(fmap), dim=-1)         # (1,h,w,D)
        sim = (e @ t[0]).squeeze(0).cpu().numpy()                       # (h,w)
        out.append(dict(file=fn, caption=cap, score=score, paper_score=paper_score, map=sim))
        print(f'{m.name[:28]:28s} {score:.3f}  {cap}', flush=True)
    del m; torch.cuda.empty_cache()
    return out


def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = [('paper', 'CLIMP-paper (CC12M)'), ('ours', 'CLIMP-ours (COCO ft)')]
    res = {k: run(k) for k, _ in models}
    fig, ax = plt.subplots(len(EXAMPLES), 1 + len(models), figsize=(4 * (1 + len(models)), 4 * len(EXAMPLES)))
    for i, (fn, cap, _) in enumerate(EXAMPLES):
        img = Image.open(f'{ROOT}/data/nocaps/val/{fn}').convert('RGB').resize((RES, RES))
        ax[i, 0].imshow(img); ax[i, 0].set_title(cap, fontsize=9, wrap=True); ax[i, 0].axis('off')
        for j, (k, label) in enumerate(models):
            r = res[k][i]; mp = r['map']
            mp = (mp - mp.min()) / (mp.max() - mp.min() + 1e-8)
            mp = np.array(Image.fromarray((mp * 255).astype(np.uint8)).resize((RES, RES), Image.BICUBIC)) / 255.
            ax[i, j + 1].imshow(img); ax[i, j + 1].imshow(mp, cmap='jet', alpha=0.5)
            ax[i, j + 1].set_title(f'{label}\nsim={r["score"]:.3f}', fontsize=9); ax[i, j + 1].axis('off')
    plt.tight_layout()
    os.makedirs(f'{ROOT}/bench/results', exist_ok=True)
    plt.savefig(f'{ROOT}/bench/results/qualitative.png', dpi=110)
    json.dump({k: [{kk: vv for kk, vv in r.items() if kk != 'map'} for r in v] for k, v in res.items()},
              open(f'{ROOT}/bench/results/qualitative.json', 'w'), indent=1)
    print('saved bench/results/qualitative.png')


if __name__ == '__main__':
    main()
