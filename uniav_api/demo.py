"""Run the function API on YouCook2 validation videos and compare with the annotations.

  python -m uniav_api.demo                        # 10 val videos from the stored features (fast)
  python -m uniav_api.demo --ids -Ju39A-G0Dk ...  # chosen val videos, stored features
  python -m uniav_api.demo --videos data/demo_videos [--cache data/demo_cache]   # raw videos, full encoders

For each video: describe_video(), then every predicted event next to the GT event it overlaps
most (IoU and GT caption). Then a few text
searches over the processed videos. Writes experiments/api_demo.json.
"""
import argparse
import json
import os
import time

import numpy as np

import uniav_api as uv
from .config import ROOT


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', default='', help='folder of raw videos (runs the ONE-PEACE encoders)')
    ap.add_argument('--features', default=os.path.join(ROOT, 'data', 'youcookii', 'av_features'))
    ap.add_argument('--ids', nargs='*', default=None, help='validation video ids (stored-feature mode)')
    ap.add_argument('--num', type=int, default=10, help='random val videos when --ids is not given')
    ap.add_argument('--device', default='auto'); ap.add_argument('--cache', default='')
    ap.add_argument('--queries', nargs='*', default=['cut the onion', 'add salt and pepper', 'fry the chicken in oil',
                                                    'put the cheese on the bread', 'boil the noodles'])
    a = ap.parse_args()
    t0 = time.time()
    pipe = uv.load(device=a.device, feature_cache=a.cache)
    print('pipeline on %s loaded in %.1f s' % (pipe.device, time.time() - t0), flush=True)
    with open(os.path.join(ROOT, 'data', 'youcookii', 'annotations', 'youcookii_annotations_trainval.json')) as f:
        db = json.load(f)['database']
    if a.videos:
        jobs = [(os.path.splitext(n)[0], os.path.join(a.videos, n)) for n in sorted(os.listdir(a.videos))
                if n.endswith(('.mp4', '.mkv', '.webm', '.mov'))]
    else:
        val = sorted(v for v, x in db.items() if x['subset'] == 'validation'
                     and os.path.exists(os.path.join(a.features, v + '_one_peace_audio.npy')))
        ids = a.ids or list(np.random.default_rng(0).choice(val, a.num, replace=False))
        jobs = [(v, None) for v in ids]
    report = []
    for vid, path in jobs:
        if path:
            r = uv.describe_video(path, video_id=vid)
        else:
            v = np.load(os.path.join(a.features, vid + '_one_peace_video_finetune.npy'))
            au = np.load(os.path.join(a.features, vid + '_one_peace_audio.npy'))
            r = uv.describe_features(v, au, db[vid]['duration'], video_id=vid)
        gts = [(x['segment'], x['sentence']) for x in db.get(vid, {}).get('annotations', [])]
        print('\n=== %s  %.0f s  %d events  (GT %d)  encode %.0f s, model %.2f s, %s'
              % (vid, r['duration'], r['num_events'], len(gts), r['timing']['encode_s'], r['timing']['model_s'], r['device']))
        hit, rows = 0, []
        for e in r['events']:
            row = {'start': e['start'], 'end': e['end'], 'score': e['score'], 'caption': e['caption']}
            if gts:
                j = int(np.argmax([iou((e['start'], e['end']), g[0]) for g in gts]))
                row.update(gt_segment=gts[j][0], gt_caption=gts[j][1], iou=round(iou((e['start'], e['end']), gts[j][0]), 2))
                hit += row['iou'] >= 0.5
            rows.append(row)
            print('  %6.1f-%6.1f  s=%.2f  %-45s | GT %-40s IoU %.2f' % (
                e['start'], e['end'], e['score'], e['caption'][:45], row.get('gt_caption', '-')[:40], row.get('iou', 0)))
        if gts:
            print('  -> %d / %d predicted events overlap a GT event with IoU >= 0.5' % (hit, len(r['events'])))
        report.append({'video_id': vid, 'duration': r['duration'], 'timing': r['timing'], 'events': rows})
    print('\n=== search')
    searches = {}
    for q in a.queries:
        hits = uv.search(q, top_k=3)
        searches[q] = hits
        print('  %-30s ' % q + ' | '.join('%s %.0f-%.0fs "%s" (%.2f)' % (h['video_id'], h['start'], h['end'], h['caption'][:30], h['score']) for h in hits))
    with open(os.path.join(ROOT, 'experiments', 'api_demo.json'), 'w') as f:
        json.dump({'videos': report, 'search': searches}, f, indent=1)


if __name__ == '__main__':
    main()
