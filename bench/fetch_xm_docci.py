import time, os
from huggingface_hub import hf_hub_download, HfApi
R = '/iitgn/home/syed.tashif/climp/data'
def get(repo, fn, local):
    for i in range(60):
        try:
            return hf_hub_download(repo, fn, repo_type='dataset', local_dir=local)
        except Exception as e:
            print('wait', repo, fn, str(e)[:80], flush=True); time.sleep(300)
    raise SystemExit('gave up ' + fn)
print(get('floschne/xm3600', 'data/en-00000-of-00001.parquet', f'{R}/xm3600/hf'), flush=True)
print('XM_OK', flush=True)
for i in range(6):
    print(get('not-lain/docci', f'data/test-0000{i}-of-00006.parquet', f'{R}/docci/hf'), flush=True)
print('DOCCI_OK', flush=True)
