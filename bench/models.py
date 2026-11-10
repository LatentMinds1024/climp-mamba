"""
Uniform wrappers for benchmarking CLIMP models with CLIP-Benchmark-style code.

Every wrapper exposes:
    .tokenizer(list[str]) -> LongTensor (B, L)  right-padded with -1 (no truncation)
    .encode_image(images) -> (B, D)  unnormalised
    .encode_text(tokens)  -> (B, D)  unnormalised
    .transform(res)       -> torchvision transform (authors' protocol: square Resize, no crop)
    .vision_backbone      -> the VMamba VSSM module (for efficiency / similarity maps)
"""
import os, sys, json, types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

ROOT = os.path.expanduser('~/climp')
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/work/VMamba/classification/models')
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MAX_TOKENS = int(os.environ.get('BENCH_MAX_TOKENS', 2048))   # effectively no truncation


def square_transform(res, mean=MEAN, std=STD):
    """Authors' eval transform (CLIMP repo data/utils.py): Resize((r, r)) -> ToTensor -> Normalize."""
    return transforms.Compose([
        transforms.Resize((res, res), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(mean, std)])


class _Tok:
    """HF tokenizer -> right-padded LongTensor with -1 as padding (mask derived inside encode_text)."""
    def __init__(self, hf_tok, max_tokens=MAX_TOKENS):
        self.tok, self.max_tokens = hf_tok, max_tokens
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        ids = self.tok(list(texts), add_special_tokens=False, truncation=True,
                       max_length=self.max_tokens)['input_ids']
        ids = [x if len(x) else [0] for x in ids]
        L = max(len(x) for x in ids)
        out = torch.full((len(ids), L), -1, dtype=torch.long)
        for i, x in enumerate(ids):
            out[i, :len(x)] = torch.tensor(x)
        return out


# ───────────────────────────── Ours: VMamba-B + in-house Mamba-1 (99M) ─────────────────────────────
class OursCLIMP(nn.Module):
    name = 'CLIMP-ours (VMamba-B + in-house Mamba-1 99M, COCO ft)'

    def __init__(self, ckpt=f'{ROOT}/runs/v3_ft/climp_ft_best.pth'):
        super().__init__()
        from climp_model import VisionMambaEncoder, TextMambaEncoder, CLIMP, build_tokenizer
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        cfg = ck.get('cfg', {})
        self.m = CLIMP(VisionMambaEncoder(None, use_checkpoint=False, log=lambda *a: None),
                       TextMambaEncoder(f'{ROOT}/ckpt/mamba-final/config.json', None, log=lambda *a: None),
                       cfg.get('EMBED_DIM', 512), vis_amp=cfg.get('VIS_AMP', 0))
        self.m.load_state_dict(ck['model'], strict=True)
        hf = build_tokenizer(f'{ROOT}/ckpt/mamba-final/tokenizer.json')
        self.pad_id = hf.pad_token_id
        self.tokenizer = _Tok(hf)
        self.vision_backbone = self.m.vis_enc.backbone
        self.eval()

    raw = False   # True: no internal autocast -> precision fully controlled by the caller (notes_eval.py)

    def encode_image(self, x):
        if self.raw:
            return self.m.vis_proj(self.m.vis_enc(x))
        return self.m.encode_image(x)

    def encode_text(self, tok):
        mask = (tok >= 0).long()
        ids = tok.clamp(min=0).masked_fill(tok < 0, self.pad_id)
        if self.raw:
            return self.m.text_proj(self.m.text_enc(ids, mask))
        return self.m.encode_text(ids, mask)

    def project_patch_features(self, fmap):
        """fmap (B,H,W,1024) after the final norm -> per-location embeddings (B,H,W,D)."""
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e = self.m.vis_proj(fmap.float())
        return e.float()


# ───────────────────────────── Authors: CLIMP-Mamba1 (CC12M) ─────────────────────────────
VMAMBA_BASE_CONFIG = dict(   # verbatim from NimrodShabtay/CLIMP models/__init__.py
    patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 15, 2], dims=128,
    ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu", ssm_conv=3,
    ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init="v2", forward_type="v05_noz",
    mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, drop_path_rate=0.6,
    patch_norm=True, norm_layer="ln2d", downsample_version="v3", patchembed_version="v2",
    gmlp=False, use_checkpoint=False, posembed=False, imgsize=224)


class AuthorsCLIMP(nn.Module):
    """Re-implementation of MambaCLIP (CLIMP repo) on official VMamba + HF transformers;
    weights = NimrodShabtay1986/CLIMP-Mamba1 model.safetensors, loaded strictly."""
    name = 'CLIMP-paper (VMamba-B + Mamba-1 1.4B, CC12M) [released ckpt]'

    def __init__(self, ckpt_dir=f'{ROOT}/ckpt/climp_authors'):
        super().__init__()
        from vmamba import VSSM
        from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
        from safetensors.torch import load_file
        self.vision_encoder = VSSM(**VMAMBA_BASE_CONFIG)
        delattr(self.vision_encoder.classifier, 'head')
        self.vision_projection = nn.Linear(1024, 768)
        tcfg = MambaConfig.from_pretrained(f'{ckpt_dir}/mamba-1.4b-hf')
        self.text_encoder = MambaForCausalLM(tcfg).backbone
        self.text_projection = nn.Linear(tcfg.hidden_size, 768)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        sd = load_file(f'{ckpt_dir}/model.safetensors')
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing[:10], unexpected[:10])
        hf = AutoTokenizer.from_pretrained(f'{ckpt_dir}/mamba-1.4b-hf')
        self.pad_id = hf.pad_token_id if hf.pad_token_id is not None else 0
        self.tokenizer = _Tok(hf)
        self.vision_backbone = self.vision_encoder
        self.eval()

    def encode_image(self, x):
        return self.vision_projection(self.vision_encoder(x))

    def encode_text(self, tok):
        mask = tok >= 0
        ids = tok.clamp(min=0).masked_fill(~mask, self.pad_id)
        h = self.text_encoder(input_ids=ids)[0]                       # (B, L, 2048)
        last = mask.long().sum(1).clamp(min=1) - 1                    # last non-pad position
        h = h[torch.arange(len(h), device=h.device), last]
        return self.text_projection(h.to(self.text_projection.weight.dtype))

    def project_patch_features(self, fmap):
        return self.vision_projection(fmap.float()).float()


# ───────────────────────────── Baseline: OpenAI CLIP ViT-B/16 ─────────────────────────────
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]


class CLIPBaseline(nn.Module):
    name = 'CLIP ViT-B/16 (OpenAI, pretrained)'
    mean, std = CLIP_MEAN, CLIP_STD

    def __init__(self, path=f'{ROOT}/ckpt/clip-vit-base-patch16'):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizerFast
        self.m = CLIPModel.from_pretrained(path, torch_dtype=torch.float32)
        hf = CLIPTokenizerFast.from_pretrained(path)
        self.pad_id = hf.pad_token_id
        self.tokenizer = _Tok(hf, max_tokens=77)          # CLIP's hard 77-token context
        self._tok = hf
        self.eval()

    def tokenize(self, texts):                            # CLIP needs BOS/EOS: use its own tokenizer
        t = self._tok(list(texts), padding=True, truncation=True, max_length=77, return_tensors='pt')
        ids = t['input_ids'].masked_fill(t['attention_mask'] == 0, -1)
        return ids

    def encode_image(self, x):
        return self.m.get_image_features(pixel_values=x, interpolate_pos_encoding=x.shape[-1] != 224)

    def encode_text(self, tok):
        mask = (tok >= 0).long()
        ids = tok.clamp(min=0).masked_fill(tok < 0, self.pad_id)
        return self.m.get_text_features(input_ids=ids, attention_mask=mask)


class SwapWrapper(nn.Module):
    """Tower-swap ablation checkpoint (ablate/train_swap.py) in the benchmark interface."""
    def __init__(self, ckpt):
        super().__init__()
        sys.path.insert(0, f'{ROOT}/ablate')
        from swap_model import load_swap
        self.m = load_swap(ckpt)
        self.name = f'swap {self.m.cfg}'
        self.pad_id = self.m.hf_tok.pad_token_id
        self.tokenizer = _Tok(self.m.hf_tok)
        self.eval()

    def encode_image(self, x):
        return self.m.encode_image(x)

    def encode_text(self, tok):
        mask = (tok >= 0).long()
        return self.m.encode_text(tok.clamp(min=0).masked_fill(tok < 0, self.pad_id), mask)


def load_model(name):
    if name.startswith('swap:'):
        return SwapWrapper(name.split(':', 1)[1])
    if name == 'clip':
        m = CLIPBaseline(); m.tokenizer = m.tokenize; return m
    if name == 'ours':
        return OursCLIMP()
    if name.startswith('ours:'):
        return OursCLIMP(name.split(':', 1)[1])
    if name == 'paper':
        return AuthorsCLIMP()
    raise ValueError(name)
