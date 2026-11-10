"""Extract NoCaps val (lmms-lab parquet mirror) -> val/*.jpg + nocaps_val.json"""
import pyarrow.parquet as pq, glob, json, os, sys
d = sys.argv[1]; os.makedirs(f'{d}/val', exist_ok=True)
out = []
for f in sorted(glob.glob(f'{d}/validation-*.parquet')):
    t = pq.read_table(f, columns=['image', 'image_file_name', 'image_id', 'annotations_captions'])
    for r in t.to_pylist():
        p = f'{d}/val/{r["image_file_name"]}'
        if not os.path.exists(p):
            open(p, 'wb').write(r['image']['bytes'])
        out.append(dict(image_id=r['image_id'], file_name=r['image_file_name'],
                        captions=[c.strip() for c in r['annotations_captions']]))
out.sort(key=lambda x: x['image_id'])
json.dump(out, open(f'{d}/nocaps_val.json', 'w'))
print(len(out), 'images', sum(len(x['captions']) for x in out), 'captions')
