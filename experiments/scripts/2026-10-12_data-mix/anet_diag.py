"""all_b16 best_seg: where ActivityNet loses R@0.5 (GT length / overlap buckets) and global NMS settings on all sets."""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2026-10-12_seg-plateau")); import ens, train
cfg = json.load(open('experiments/runs/b8a86a5e6ef7-all_b16.json'))['config']
cfg['coin']['train'] = False; cfg['num_workers'] = 4; cfg['batch_size'] = 16
for k in cfg['extra_train']: cfg['extra_train'][k]['ratio'] = 0.0
ds, dv, dl, dlv, extra, ov = train.build_loaders(cfg, train.fix_random_seed(1, include_cuda=True))
model, _ = ens.load('ckpt/all_b16/best_seg.pth.tar', 'experiments/runs/b8a86a5e6ef7-all_b16.json')
sets = dict({'yc2': dlv}, **{n: l for n, (l, d) in ov.items()})

def run(loader):
    out = []
    with torch.no_grad():
        for batch in loader:
            for x in batch: x['feats'] = {k: v.to(ens.dev) for k, v in x['feats'].items()}
            for r, x in zip(model(batch), batch):
                out.append((np.asarray(r['gt_segments'], float), r['segments'].numpy(), float(x['duration'])))
    return out

def iou1(p, g):   # (P,2) vs (2,)
    inter = np.clip(np.minimum(p[:, 1], g[1]) - np.maximum(p[:, 0], g[0]), 0, None)
    return inter / (np.maximum(p[:, 1], g[1]) - np.minimum(p[:, 0], g[0]) + 1e-9)

def r05(res):
    h = n = 0
    for g, P, d in res:
        k = len(g)
        for gg in g:
            n += 1; h += len(P) and iou1(P[:k], gg).max() >= .5
    return 100 * h / n

# 1. ActivityNet breakdown at the default setting
res = run(sets['anet'])
rows = []
for g, P, d in res:
    k = len(g)
    for i, gg in enumerate(g):
        others = np.delete(g, i, 0)
        ovl = len(others) and iou1(others, gg).max() >= .3
        v = iou1(P[:k], gg) if len(P) else np.zeros(1)
        rows.append(((gg[1] - gg[0]) / d, ovl, v.max() >= .5, len(P) and iou1(P, gg).max() >= .5, k))
rows = np.array(rows, float)
print('ANet R@0.5 %.1f over %d GT' % (100 * rows[:, 2].mean(), len(rows)))
for lo, hi in [(0, .1), (.1, .25), (.25, .5), (.5, .9), (.9, 1.01)]:
    m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    print('  GT length %3.0f-%3.0f%% of video: %4.1f%% of GT | R@0.5 at K %.1f, top-100 %.1f' % (100 * lo, 100 * hi, 100 * m.mean(), 100 * rows[m, 2].mean(), 100 * rows[m, 3].mean()))
for o in (0, 1):
    m = rows[:, 1] == o
    print('  %s another GT (IoU>=0.3): %4.1f%% of GT | R@0.5 at K %.1f, top-100 %.1f' % ('overlaps' if o else 'no overlap with', 100 * m.mean(), 100 * rows[m, 2].mean(), 100 * rows[m, 3].mean()))
pl = np.concatenate([(P[:len(g), 1] - P[:len(g), 0]) / d for g, P, d in res])
gl = rows[:, 0]
print('  predicted length (top K) %% of video: p25 %.0f median %.0f p75 %.0f | GT p25 %.0f median %.0f p75 %.0f' % (*np.percentile(100 * pl, [25, 50, 75]), *np.percentile(100 * gl, [25, 50, 75])))

# 2. one global NMS setting for every set
for soft, iou, sigma in [(True, .7, .5), (True, .7, .75), (True, .7, 1.0), (True, .7, 2.0), (False, .9, .5), (False, .7, .5)]:
    model.nms_cfg.update(soft=soft, iou=iou, sigma=sigma)
    r = {n: r05(run(l)) for n, l in sets.items()}
    print('NMS soft=%d iou=%.2f sigma=%.2f | %s | avg %.2f' % (soft, iou, sigma, ' '.join('%s %.2f' % kv for kv in r.items()), np.mean(list(r.values()))), flush=True)
