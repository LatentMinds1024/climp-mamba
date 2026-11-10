"""Shared CLIMP model / data / metric code (used by train_climp_ft.py and eval_climp.py)."""
import os, sys, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT = os.path.expanduser('~/climp')
VMAMBA_DIR = os.environ.get('VMAMBA_DIR', f'{ROOT}/work/VMamba/classification/models')
sys.path.insert(0, VMAMBA_DIR)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
val_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


# ───────────────────────────── Vision tower: VMamba-B ─────────────────────────────
class VisionMambaEncoder(nn.Module):
    def __init__(self, ckpt_path=None, use_checkpoint=True, log=print):
        super().__init__()
        from vmamba import VSSM
        self.backbone = VSSM(
            in_chans=3, patch_size=4, num_classes=1000,
            dims=[128, 256, 512, 1024], depths=[2, 2, 15, 2],
            ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank='auto',
            ssm_act_layer='silu', ssm_conv=3, ssm_conv_bias=False,
            ssm_drop_rate=0.0, ssm_init='v0', forward_type='v05_noz',
            mlp_ratio=4.0, mlp_act_layer='gelu', mlp_drop_rate=0.0,
            gmlp=False, drop_path_rate=0.1, patch_norm=True,
            patchembed_version='v2', downsample_version='v3',
            use_checkpoint=bool(use_checkpoint),
        )
        if ckpt_path:
            ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state = ckpt.get('model', ckpt)
            ms    = self.backbone.state_dict()
            matched = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
            missing, _ = self.backbone.load_state_dict(matched, strict=False)
            log(f'VMamba-B loaded  matched={len(matched)}/{len(state)}  missing={len(missing)} {missing[:5]}')
        self.backbone.classifier.head = nn.Identity()
        self.out_dim = 1024

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() == 4:   feats = feats.mean(dim=(1, 2))
        elif feats.dim() == 3: feats = feats.mean(dim=1)
        return feats  # (B, 1024)


# ───────────────────────────── Text tower: in-house Mamba-1 ─────────────────────────────
class TextMambaEncoder(nn.Module):
    """Exactly the backbone of the in-house MambaLMHeadModel (RMSNorm, fused add-norm,
    residual in fp32). Masked mean pooling over non-pad tokens."""
    def __init__(self, cfg_path, ckpt_path=None, log=print):
        super().__init__()
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MixerModel
        c = MambaConfig(**json.load(open(cfg_path)))
        vocab = c.vocab_size
        if vocab % c.pad_vocab_size_multiple:
            vocab += c.pad_vocab_size_multiple - vocab % c.pad_vocab_size_multiple
        self.backbone = MixerModel(
            d_model=c.d_model, n_layer=c.n_layer, d_intermediate=c.d_intermediate,
            vocab_size=vocab, ssm_cfg=c.ssm_cfg, attn_layer_idx=c.attn_layer_idx,
            attn_cfg=c.attn_cfg, rms_norm=c.rms_norm, initializer_cfg=None,
            fused_add_norm=c.fused_add_norm, residual_in_fp32=c.residual_in_fp32,
        )
        if ckpt_path:
            raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd  = {k[len('backbone.'):]: v for k, v in raw['model'].items() if k.startswith('backbone.')}
            self.backbone.load_state_dict(sd, strict=True)
            log(f'Text Mamba loaded strictly: {len(sd)} tensors  (pretrain step={raw.get("step")}, '
                f'val_ppl={raw.get("val_ppl", float("nan")):.2f})')
        self.out_dim = c.d_model

    def forward(self, input_ids, attn_mask):
        h = self.backbone(input_ids)                          # (B, L, D)
        m = attn_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)       # (B, D)


def build_tokenizer(tokenizer_file):
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file,
                                  bos_token='<|endoftext|>', eos_token='<|endoftext|>',
                                  unk_token='<|endoftext|>', pad_token='<|padding|>')
    tok.padding_side = 'right'
    return tok


# ───────────────────────────── CLIMP ─────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(in_d, in_d), nn.GELU(), nn.Linear(in_d, out_d))
        self.norm = nn.LayerNorm(out_d)
    def forward(self, x): return self.norm(self.net(x))


class CLIMP(nn.Module):
    def __init__(self, vis_enc, text_enc, embed_dim=512, vis_amp=False):
        super().__init__()
        self.vis_enc, self.text_enc = vis_enc, text_enc
        self.vis_proj  = ProjectionHead(vis_enc.out_dim,  embed_dim)
        self.text_proj = ProjectionHead(text_enc.out_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.vis_amp = bool(vis_amp)

    def encode_image(self, images):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.vis_amp):
            f = self.vis_enc(images)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e = self.vis_proj(f.float())
        return F.normalize(e.float(), dim=-1)

    def encode_text(self, ids, mask):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e = self.text_proj(self.text_enc(ids, mask))
        return F.normalize(e.float(), dim=-1)

    def forward(self, images, ids, mask):
        return self.encode_image(images), self.encode_text(ids, mask), \
               self.logit_scale.exp().clamp(max=100)


# ───────────────────────────── Retrieval evaluation ─────────────────────────────
class ImageFiles(Dataset):
    def __init__(self, img_dir, files): self.img_dir, self.files = img_dir, files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        return val_tf(Image.open(os.path.join(self.img_dir, self.files[i])).convert('RGB'))


def load_coco5k(coco_root):
    """COCO val2017 = the standard 5K split. Returns (img_dir, files, captions, cap2img)."""
    v = json.load(open(f'{coco_root}/annotations/captions_val2017.json'))
    id2file = {x['id']: x['file_name'] for x in v['images']}
    ids = sorted({a['image_id'] for a in v['annotations']})
    idx = {iid: i for i, iid in enumerate(ids)}
    caps = [a['caption'].strip() for a in v['annotations']]
    return f'{coco_root}/val2017', [id2file[i] for i in ids], caps, \
           torch.tensor([idx[a['image_id']] for a in v['annotations']])


def load_nocaps_val(nocaps_root):
    """NoCaps val: 4500 images x 10 captions. Returns (img_dir, files, captions, cap2img)."""
    d = json.load(open(f'{nocaps_root}/nocaps_val.json'))
    files, caps, c2i = [], [], []
    for i, x in enumerate(d):
        files.append(x['file_name'])
        for c in x['captions']:
            caps.append(c); c2i.append(i)
    return f'{nocaps_root}/val', files, caps, torch.tensor(c2i)


@torch.no_grad()
def retrieval_eval(model, tokenizer, img_dir, files, caps, cap2img, device,
                   max_len=77, ks=(1, 5, 10), num_workers=8, img_bs=128, txt_bs=512,
                   mp_context=None):
    was_training = model.training
    model.eval()
    # mp_context='spawn' avoids forked workers segfaulting after the parent has run threaded
    # CPU work (seen on the 2nd dataset of a standalone eval). Requires a __main__ guard.
    dl = DataLoader(ImageFiles(img_dir, files), batch_size=img_bs,
                    num_workers=num_workers, pin_memory=True,
                    multiprocessing_context=mp_context if num_workers > 0 else None)
    img = torch.cat([model.encode_image(x.to(device, non_blocking=True)) for x in dl])
    txt = []
    for i in range(0, len(caps), txt_bs):
        t = tokenizer(caps[i:i + txt_bs], max_length=max_len, padding='max_length',
                      truncation=True, return_tensors='pt')
        txt.append(model.encode_text(t['input_ids'].to(device), t['attention_mask'].to(device)))
    txt = torch.cat(txt)
    sims = img @ txt.T                                       # (N_img, N_cap)
    gt = cap2img.to(device)
    K = max(ks)
    i2t_hit = gt[sims.topk(K, dim=1).indices] == torch.arange(len(img), device=device)[:, None]
    t2i_hit = sims.T.topk(K, dim=1).indices == gt[:, None]
    res, rsum = {}, 0.0
    for k in ks:
        res[f'I2T_R@{k}'] = i2t_hit[:, :k].any(1).float().mean().item() * 100
        res[f'T2I_R@{k}'] = t2i_hit[:, :k].any(1).float().mean().item() * 100
    for k in (1, 5, 10):
        if k in ks: rsum += res[f'I2T_R@{k}'] + res[f'T2I_R@{k}']
    res['Rsum'] = rsum
    res['n_images'], res['n_captions'] = len(img), len(caps)
    if was_training: model.train()
    return res
