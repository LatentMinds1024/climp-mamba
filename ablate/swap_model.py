"""
Tower-swap CLIMP: any vision tower x any text tower, each frozen / partially / fully trainable.

VIS  = paper   -> VMamba-B with the CLIMP-paper CC12M weights (NimrodShabtay1986/CLIMP-Mamba1)
       inet    -> VMamba-B ImageNet-1k init (ckpt/vssm_base_ckpt.pth), same as our v3 fine-tune
       ours    -> VMamba-B from our COCO fine-tune (runs/v3_ft/climp_ft_best.pth)
TXT  = inhouse -> in-house Mamba-1 99M (2B tokens), mamba_ssm MixerModel
       paper   -> Mamba-1 1.4B text tower from the CLIMP-paper checkpoint
       hf:<dir> -> any HF MambaForCausalLM dir (e.g. ckpt/hf/mamba-130m-hf, -790m-hf, -2.8b-hf)
TRAIN_VIS / TRAIN_TXT: -1 = all layers, 0 = frozen, N = last N blocks (+ final norm) trainable.
Text pooling: masked mean over tokens (same for every text tower).
"""
import os, sys, json, math
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.expanduser('~/climp')
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/work/VMamba/classification/models')
sys.path.insert(0, f'{ROOT}/bench')
from climp_model import ProjectionHead, TextMambaEncoder, VisionMambaEncoder, build_tokenizer

PAPER_DIR = f'{ROOT}/ckpt/climp_authors'


def _paper_sd(prefix):
    from safetensors.torch import load_file
    sd = load_file(f'{PAPER_DIR}/model.safetensors')
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class PaperVision(nn.Module):
    def __init__(self, grad_ckpt=False):
        super().__init__()
        from vmamba import VSSM
        from models import VMAMBA_BASE_CONFIG
        cfg = dict(VMAMBA_BASE_CONFIG, use_checkpoint=bool(grad_ckpt), drop_path_rate=0.1)
        self.backbone = VSSM(**cfg)
        delattr(self.backbone.classifier, 'head')
        sd = _paper_sd('vision_encoder.')
        self.backbone.load_state_dict(sd, strict=True)
        self.out_dim = 1024

    def forward(self, x):
        return self.backbone(x)


class HFMambaText(nn.Module):
    def __init__(self, src):
        super().__init__()
        from transformers import MambaConfig, MambaForCausalLM
        if src == 'paper':
            cfg = MambaConfig.from_pretrained(f'{PAPER_DIR}/mamba-1.4b-hf')
            self.backbone = MambaForCausalLM(cfg).backbone
            self.backbone.load_state_dict(_paper_sd('text_encoder.'), strict=True)
            self.tok_dir = f'{PAPER_DIR}/mamba-1.4b-hf'
        else:
            self.backbone = MambaForCausalLM.from_pretrained(src, torch_dtype=torch.float32).backbone
            self.tok_dir = src
        self.out_dim = self.backbone.config.hidden_size

    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


def build_vision(name, grad_ckpt):
    if name == 'paper':
        return PaperVision(grad_ckpt)
    if name == 'inet':
        return VisionMambaEncoder(f'{ROOT}/ckpt/vssm_base_ckpt.pth', grad_ckpt, log=lambda *a: None)
    if name == 'ours':
        v = VisionMambaEncoder(None, grad_ckpt, log=lambda *a: None)
        sd = torch.load(f'{ROOT}/runs/v3_ft/climp_ft_best.pth', map_location='cpu', weights_only=False)['model']
        v.load_state_dict({k[len('vis_enc.'):]: t for k, t in sd.items() if k.startswith('vis_enc.')}, strict=True)
        return v
    raise ValueError(name)


def build_text(name):
    """Returns (tower, hf_tokenizer)."""
    if name == 'inhouse':
        t = TextMambaEncoder(f'{ROOT}/ckpt/mamba-final/config.json', f'{ROOT}/ckpt/ckpt/best_model.pt',
                             log=lambda *a: None)
        return t, build_tokenizer(f'{ROOT}/ckpt/mamba-final/tokenizer.json')
    from transformers import AutoTokenizer
    t = HFMambaText('paper' if name == 'paper' else name.split(':', 1)[1])
    tok = AutoTokenizer.from_pretrained(t.tok_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    return t, tok


def _blocks(tower):
    b = tower.backbone
    if hasattr(b, 'norm_f'):                                                 # text (MixerModel / HF Mamba)
        return list(b.layers), [b.norm_f]
    blocks = [blk for layer in b.layers for blk in layer.blocks]             # VMamba: flatten stages
    return blocks, [b.classifier]


def set_trainable(tower, n):
    """n = -1 all, 0 none, N = last N blocks + final norm."""
    for p in tower.parameters():
        p.requires_grad = n < 0
    if n > 0:
        blocks, tail = _blocks(tower)
        for m in blocks[-n:] + tail:
            for p in m.parameters():
                p.requires_grad = True
    return sum(p.numel() for p in tower.parameters() if p.requires_grad)


class SwapCLIMP(nn.Module):
    def __init__(self, vis, txt, train_vis=0, train_txt=-1, embed_dim=512, grad_ckpt=True):
        super().__init__()
        self.cfg = dict(vis=vis, txt=txt, train_vis=train_vis, train_txt=train_txt, embed_dim=embed_dim)
        self.vis_enc = build_vision(vis, grad_ckpt and train_vis != 0)
        self.text_enc, self.hf_tok = build_text(txt)
        self.vis_proj = ProjectionHead(self.vis_enc.out_dim, embed_dim)
        self.text_proj = ProjectionHead(self.text_enc.out_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.n_train_vis = set_trainable(self.vis_enc, train_vis)
        self.n_train_txt = set_trainable(self.text_enc, train_txt)
        if train_vis == 0:
            self.vis_enc.eval()
        if train_txt == 0:
            self.text_enc.eval()
        if train_txt == 0 and txt != 'inhouse':   # frozen big towers: keep weights in bf16
            self.text_enc.to(torch.bfloat16)

    def train(self, mode=True):           # frozen towers stay in eval mode (no drop-path)
        super().train(mode)
        if self.cfg['train_vis'] == 0: self.vis_enc.eval()
        if self.cfg['train_txt'] == 0: self.text_enc.eval()
        return self

    def tokenize(self, caps, max_len=77):
        t = self.hf_tok(list(caps), max_length=max_len, padding=True, truncation=True, return_tensors='pt')
        return t['input_ids'], t['attention_mask']

    vis_amp = False   # set True by the trainer to run a *trainable* vision tower in bf16 autocast

    def encode_image(self, x):
        frozen = self.cfg['train_vis'] == 0
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen), \
             torch.autocast('cuda', dtype=torch.bfloat16, enabled=frozen or self.vis_amp):
            f = self.vis_enc(x)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e = self.vis_proj(f.float())
        return F.normalize(e.float(), dim=-1)

    def encode_text(self, ids, mask):
        frozen = self.cfg['train_txt'] == 0
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen), \
             torch.autocast('cuda', dtype=torch.bfloat16):
            f = self.text_enc(ids, mask)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e = self.text_proj(f.float())
        return F.normalize(e.float(), dim=-1)

    def forward(self, images, ids, mask):
        return self.encode_image(images), self.encode_text(ids, mask), self.logit_scale.exp().clamp(max=100)

    def trainable_state_dict(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in keep or k.startswith(('vis_proj.', 'text_proj.')) or k == 'logit_scale'}


def load_swap(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    m = SwapCLIMP(**ck['swap_cfg'], grad_ckpt=False)
    load_partial(m, ck['model'])
    return m


def load_partial(m, part):
    """VMamba's Linear2d cannot load a partial state dict -> merge onto the full one, load strictly."""
    full = m.state_dict()
    unexpected = [k for k in part if k not in full]
    assert not unexpected, unexpected[:5]
    full.update({k: v.to(full[k].dtype) for k, v in part.items()})
    m.load_state_dict(full, strict=True)
