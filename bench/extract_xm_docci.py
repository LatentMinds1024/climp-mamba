import pyarrow.parquet as pq, glob, os, json, sys
R = '/iitgn/home/syed.tashif/climp/data'
if 'xm' in sys.argv:
    os.makedirs(f'{R}/xm3600/images', exist_ok=True)
    t = pq.read_table(f'{R}/xm3600/hf/data/en-00000-of-00001.parquet')
    with open(f'{R}/xm3600/captions_en.jsonl', 'w') as f:
        for r in t.to_pylist():
            open(f'{R}/xm3600/images/{r["image_id"]}.jpg', 'wb').write(r['image']['bytes'])
            f.write(json.dumps({'image/key': r['image_id'], 'en': {'caption': r['captions']}}) + '\n')
    print('xm3600', t.num_rows, 'images')
if 'docci' in sys.argv:
    # mirror (not-lain/docci) has image/question/answer; answer == official description -> match on it
    os.makedirs(f'{R}/docci/images', exist_ok=True)
    test = {}
    for l in open(f'{R}/docci/docci_descriptions.jsonlines'):
        r = json.loads(l)
        if r['split'] == 'test':
            test[r['description'].strip()] = r
    n, seen = 0, set()
    for fn in sorted(glob.glob(f'{R}/docci/hf/data/test-*.parquet')):
        for b in pq.ParquetFile(fn).iter_batches(batch_size=64, columns=['image', 'answer']):
            for r in b.to_pylist():
                rec = test[r['answer'].strip()]
                open(f'{R}/docci/images/{rec["image_file"]}', 'wb').write(r['image']['bytes'])
                seen.add(rec['example_id']); n += 1
    print('docci test images written', n, 'unique', len(seen), 'of', len(test))
