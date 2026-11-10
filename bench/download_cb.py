"""Download CLIP-Benchmark webdataset test splits used in CLIMP paper Tables 9/10."""
import os, sys, time
from huggingface_hub import snapshot_download
DS = ['vtab-cifar10','vtab-cifar100','fer2013','vtab-smallnorb_label_azimuth','vtab-smallnorb_label_elevation',
      'vtab-svhn','flickr8k','flickr30k','imagenet-o','vtab-dsprites_label_orientation','vtab-dsprites_label_x_position',
      'vtab-dsprites_label_y_position','voc2007','imagenet-a','mscoco_captions','fgvc_aircraft','cars','imagenetv2',
      'vtab-clevr_closest_object_distance','vtab-clevr_count_all','vtab-dmlab','imagenet-r','country211','imagenet1k',
      'imagenet_sketch','vtab-diabetic_retinopathy','sun397','objectnet']
root = os.path.expanduser('~/climp/data/clip_benchmark')
for d in DS:
    t0 = time.time()
    for attempt in range(5):
        try:
            snapshot_download(f'clip-benchmark/wds_{d}', repo_type='dataset', local_dir=f'{root}/wds_{d}',
                              allow_patterns=['test/*', '*.txt'], max_workers=8)
            break
        except Exception as e:
            print('retry', d, e, flush=True); time.sleep(30)
    print(f'DONE {d} {time.time()-t0:.0f}s', flush=True)
print('ALLDONE', flush=True)
