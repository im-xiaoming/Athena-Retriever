"""Check the inference copy against training-time evaluation and pick the event threshold.

  python -m uniav_api.calibrate

Uses the stored validation features (data/youcookii/iv2_feats, or iv2_dense for a 2 rows/s model),
so no video encoding.
1. Parity: with k = number of GT events per video, R@IoU must match train_event.py --eval.
2. Threshold: in production the number of events is unknown, so events are kept by score.
   For each min_score, precision / recall / F1 at IoU 0.5 (one-to-one greedy matching);
   prints the best F1 setting to put in config.py.
"""
import json
import os

import numpy as np

from .config import ROOT, Config
from .pipeline import UniAVPipeline, select_events


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr=0.5):
    used, tp = set(), 0
    for p in preds:
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if j not in used and iou(p, g) > best:
                best, bj = iou(p, g), j
        if best >= thr:
            used.add(bj); tp += 1
    return tp


def main():
    pipe = UniAVPipeline(Config.from_env())
    print('checkpoint %s (InternVideo2 %s, caption space %s, %s segmentation weights)' % (
        pipe.cfg.checkpoint, '+'.join(pipe.spec.video_keys), pipe.caption_space,
        'separate' if pipe.seg_model is not pipe.model else 'the same'))
    with open(os.path.join(ROOT, 'data', 'youcookii', 'annotations', 'youcookii_annotations_trainval.json')) as f:
        db = json.load(f)['database']
    F = os.path.join(ROOT, 'data', 'youcookii', 'iv2_feats' if pipe.spec.stride == pipe.spec.fps else 'iv2_dense')
    have = set(pipe.spec.stored_ids(F))
    cands, n_gt_total = [], 0
    for vid, x in sorted(db.items()):
        if x['subset'] != 'validation' or vid not in have:
            continue
        fv, fa, n = pipe.spec.prepare(*pipe.spec.from_store(vid, F))
        segs, scores, _ = pipe.seg_model(fv.to(pipe.device), fa.to(pipe.device))
        secs = pipe.spec.to_seconds(segs, n, x['duration'])
        gts = [a['segment'] for a in x['annotations'][:16]]
        cands.append((secs, scores, gts)); n_gt_total += len(gts)
    print('validation videos: %d, GT events: %d' % (len(cands), n_gt_total))

    # 1. parity with training evaluation (k = #GT, best IoU per GT)
    cov = {0.3: 0, 0.5: 0, 0.7: 0}
    for secs, scores, gts in cands:
        k = min(len(gts), len(secs))
        for g in gts:
            best = max([iou(p, g) for p in secs[:k]] or [0.0])
            for t in cov:
                cov[t] += best >= t
    print('parity (k = #GT): ' + '  '.join('R@%.1f %.2f' % (t, 100.0 * c / n_gt_total) for t, c in cov.items()))

    # 2. threshold sweep for production selection
    cfg = pipe.cfg
    print('%11s %9s %8s %8s %8s %8s' % ('max_overlap', 'min_score', 'events/v', 'prec', 'recall', 'F1'))
    best = None
    for mo in (0.1, 0.2, 0.3, 0.5):
        for ms in np.arange(0.20, 0.76, 0.04):
            tp = n_pred = 0
            for secs, scores, gts in cands:
                keep = select_events(secs, scores, ms, mo, cfg.max_events)
                n_pred += len(keep)
                tp += match([secs[i] for i in keep], gts)
            p, r = tp / max(n_pred, 1), tp / n_gt_total
            f1 = 2 * p * r / max(p + r, 1e-9)
            print('%11.1f %9.2f %8.2f %8.3f %8.3f %8.3f' % (mo, ms, n_pred / len(cands), p, r, f1))
            if best is None or f1 > best[0]:
                best = (f1, ms, mo, n_pred / len(cands), p, r)
    print('best: min_score %.2f max_overlap %.1f -> F1 %.3f (precision %.3f, recall %.3f), %.1f events per video'
          ' (GT mean %.1f)' % (best[1], best[2], best[0], best[4], best[5], best[3], n_gt_total / len(cands)))


if __name__ == '__main__':
    main()
