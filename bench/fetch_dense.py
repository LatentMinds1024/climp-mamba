import time
from huggingface_hub import hf_hub_download
R = '/iitgn/home/syed.tashif/climp/data'
def get(repo, fn, local):
    for i in range(60):
        try:
            p = hf_hub_download(repo, fn, repo_type='dataset', local_dir=local); print('OK', p, flush=True); return p
        except Exception as e:
            print('wait', repo, fn, str(e)[:100], flush=True); time.sleep(300)
get('BeichenZhang/Urban1k', 'Urban1k.zip', f'{R}/urban1k')
get('Lin-Chen/ShareGPT4V', 'share-captioner_coco_lcs_sam_1246k_1107.json', f'{R}/dense/sharegpt4v')
get('CaptionEmporium/conceptual-captions-cc12m-llavanext', 'train.jsonl.gz', f'{R}/dense/cc12m_llavanext')
get('liuhaotian/LLaVA-Pretrain', 'images.zip', f'{R}/dense/llava_pretrain')
print('DENSE_DONE', flush=True)
