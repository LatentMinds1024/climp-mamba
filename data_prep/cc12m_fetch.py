"""Download pixparse/cc12m-wds (paper's training set) + build key->captions SQLite from LLaVA-NeXT recaptions."""
import os, sys, gzip, json, sqlite3, time, threading
from huggingface_hub import snapshot_download
ROOT = os.path.expanduser('~/climp/data')

def build_db():
    db_path = f'{ROOT}/cc12m/captions.sqlite'
    if os.path.exists(db_path + '.done'):
        return
    if os.path.exists(db_path): os.remove(db_path)
    con = sqlite3.connect(db_path); con.execute('CREATE TABLE c (k TEXT PRIMARY KEY, alt TEXT, short TEXT, long TEXT)')
    rows, n = [], 0
    for line in gzip.open(f'{ROOT}/dense/cc12m_llavanext/train.jsonl.gz', 'rt'):
        d = json.loads(line)
        rows.append((d['key'], d.get('caption') or '', (d.get('caption_llava_short') or '').strip(), (d.get('caption_llava') or '').strip()))
        if len(rows) == 200000:
            con.executemany('INSERT OR REPLACE INTO c VALUES (?,?,?,?)', rows); con.commit(); n += len(rows); rows = []
            print(f'[db] {n:,}', flush=True)
    con.executemany('INSERT OR REPLACE INTO c VALUES (?,?,?,?)', rows); con.commit(); n += len(rows)
    con.close(); open(db_path + '.done', 'w').write(str(n)); print(f'[db] done {n:,}', flush=True)

t = threading.Thread(target=build_db); t.start()
for attempt in range(50):
    try:
        snapshot_download('pixparse/cc12m-wds', repo_type='dataset', local_dir=f'{ROOT}/cc12m',
                          allow_patterns=['*.tar', '_info.json'], max_workers=int(os.environ.get('DL_WORKERS', 8)))
        break
    except Exception as e:
        print(f'[dl] attempt {attempt} failed: {str(e)[:200]} -> retry in 120s', flush=True); time.sleep(120)
t.join()
n = len([f for f in os.listdir(f'{ROOT}/cc12m') if f.endswith('.tar')])
print(f'[dl] FETCH_DONE {n} tars', flush=True)
