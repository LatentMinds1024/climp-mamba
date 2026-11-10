"""Image-text pair datasets for contrastive training. DATA = comma-separated list of sources."""
import os, json, random
from torch.utils.data import Dataset
from PIL import Image

ROOT = os.path.expanduser('~/climp')


def coco_pairs():
    a = json.load(open(f'{ROOT}/data/coco/annotations/captions_train2017.json'))
    id2f = {x['id']: x['file_name'] for x in a['images']}
    caps = {}
    for x in a['annotations']:
        caps.setdefault(x['image_id'], []).append(x['caption'].strip())
    return [(f'{ROOT}/data/coco/train2017/{id2f[i]}', c) for i, c in caps.items()]


SOURCES = {'coco': coco_pairs}


class PairDataset(Dataset):
    """Items: (image_path, [captions]); one caption sampled per access."""
    def __init__(self, data, transform):
        self.items = [it for s in data.split(',') for it in SOURCES[s]()]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, caps = self.items[i]
        return self.transform(Image.open(p).convert('RGB')), random.choice(caps)


# ───────────────────────── CC12M webdataset (pixparse/cc12m-wds) + LLaVA-NeXT recaptions ─────────────────────────
import glob, io, sqlite3
import torch

CC12M_DIR = f'{ROOT}/data/cc12m'
CC12M_N = 10_968_539


def cc12m_shards(spec=''):
    """spec '' = all downloaded tars; 'a:b' = slice of the sorted list; '@file' = list in a file."""
    if spec.startswith('@'):
        return [l.strip() for l in open(spec[1:]) if l.strip()]
    tars = sorted(glob.glob(f'{CC12M_DIR}/cc12m-train-*.tar'))
    if ':' in spec:
        a, b = spec.split(':'); tars = tars[int(a or 0):int(b) if b else None]
    return tars


class _Caps:
    """Lazy per-worker SQLite reader: key -> (alt, short, long)."""
    def __init__(self):
        self.con = None
    def get(self, k):
        if self.con is None:
            self.con = sqlite3.connect(f'file:{CC12M_DIR}/captions.sqlite?mode=ro&immutable=1', uri=True,
                                       check_same_thread=False)
        r = self.con.execute('SELECT alt, short, long FROM c WHERE k=?', (k,)).fetchone()
        return r


def pick_caption(alt, rec, mode):
    """mode: alt = original CC12M alt-text (paper-faithful); short / long = LLaVA-NeXT recaptions;
       mix = uniform over {alt, short, long} (whatever is non-empty)."""
    a, s, l = (rec if rec else ('', '', ''))
    a = a or alt
    if mode == 'alt': c = [a]
    elif mode == 'short': c = [s or a]
    elif mode == 'long': c = [l or s or a]
    else: c = [x for x in (a, s, l) if x] or [alt]
    return random.choice(c)


def cc12m_stream(transform, shards, cap_mode='alt', seed=0):
    """Infinite resampled stream (open_clip style): each rank/worker samples shards with replacement,
       so every rank always has data (no DDP hang on uneven shards)."""
    import webdataset as wds
    caps = _Caps()

    def to_pair(s):
        alt = s.get('txt', b'').decode('utf-8', 'ignore') if isinstance(s.get('txt'), bytes) else s.get('txt', '')
        img = Image.open(io.BytesIO(s['jpg'])).convert('RGB')
        return transform(img), pick_caption(alt, caps.get(s['__key__']), cap_mode)

    return (wds.WebDataset(shards, resampled=True, handler=wds.warn_and_continue, nodesplitter=None,
                           shardshuffle=False, empty_check=False, seed=None)
            .shuffle(5000, handler=wds.warn_and_continue)
            .map(to_pair, handler=wds.warn_and_continue))
