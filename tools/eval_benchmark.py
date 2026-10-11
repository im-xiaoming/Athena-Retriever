"""Benchmark evaluation shared by the original UniAV and our model: class-agnostic temporal localization.

  python tools/eval_benchmark.py predict --ckpt ckpt/all_b16/best_seg.pth.tar --run experiments/runs/<host>-all_b16.json
  python tools/eval_benchmark.py predict --ckpt ckpt/api/athena.pth             # API checkpoint: its config is inside
  python tools/eval_benchmark.py score [--sets dcase unav100]                    # every file in benchmark/preds/

Sets: DCASE public_eval (564 clips) and UnAV-100 test (2167 videos), annotations in benchmark/<folder>/<set>_all.json.
Inputs: tools/benchmark_data.py fetch; UniAV predictions: tools/eval_uniav.py.

The two models do not share a task: UniAV names every event with a class, ours segments and captions. Scored is
what both do, finding the events, so class labels are dropped on both sides:
  - GT: events with the same span are one event
  - predictions: identical spans (UniAV gives one per class) are one prediction with the highest score
  - AP@t: ActivityNet average precision (all videos pooled, best score first, interpolated) at tIoU t, with
    UniAV's thresholds 0.1:0.9; mAP is the mean over them
  - R@t / mIoU: as train.py, each video keeps its k best predictions (k = its number of GT events);
    R@t is the share of GT events whose best IoU with them is >= t, mIoU the mean of that best IoU
A video without predictions misses all its events.
Video sets: "common" holds the videos every scored model has features for (DCASE: some videos could not be
downloaded for our features, UnAV-100: features for part of the set only); "all" is each model's own set.

Prediction files (benchmark/preds/<model>_<set>.json):
  {"model": ..., "set": ..., "videos": [every evaluated id], "results": {id: [[start, end, score, ...], ...]}}
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {'dcase': ('desed', 'public_eval'), 'unav100': ('unav100', 'test')}   # set -> (folder in benchmark/, subset)
TIOUS = np.round(np.linspace(0.1, 0.9, 9), 2)
PREDS = os.path.join(ROOT, 'benchmark', 'preds')


def anno_file(name):
    return os.path.join(ROOT, 'benchmark', SETS[name][0], '%s_all.json' % name)


# ======================================= PREDICT (our model) =======================================================
def merge(base, over):
    """base updated with over, recursively for dicts."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def predict(a):
    import torch
    import yaml
    sys.path.insert(0, ROOT)
    from libs.datasets.benchmark import BenchmarkDataset
    from libs.datasets.data_utils import trivial_batch_collator
    from libs.modeling.blocks import upgrade_state_dict
    from libs.modeling.event_archs import EventCaptionTransformer
    from train import load_ckpt

    with open(os.path.join(ROOT, 'configs', 'youcook2_event.yaml')) as f:
        cfg = yaml.safe_load(f)
    if not hasattr(np, '_core'):   # numpy < 2 (training env) reading numbers pickled by numpy 2 (Colab, API export)
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    ck = load_ckpt(a.ckpt)
    if 'config' in ck:   # API checkpoint (tools/export_api_ckpt.py): segmentation weights if stored apart
        cfg, sd = merge(cfg, ck['config']), ck.get('state_dict_seg', ck['state_dict'])
    else:                # training checkpoint: the config of its run record
        assert a.run, '--run experiments/runs/<host>-<run>.json is needed with a training checkpoint'
        with open(a.run) as f:
            cfg = merge(cfg, json.load(f)['config'])
        sd = ck['state_dict']
    sd = upgrade_state_dict({k: v.float() for k, v in sd.items()})
    name = a.name or ck.get('run') or os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    os.makedirs(PREDS, exist_ok=True)
    for s in a.sets:
        folder, subset = SETS[s]
        ds = BenchmarkDataset(subset, anno_file(s), os.path.join(ROOT, 'data', s, 'iv2_feats'), **cfg['dataset'])
        if model is None:
            mcfg = dict(cfg['model'], input_dim_V=ds.feat_dims[0], input_dim_A=ds.feat_dims[1],
                        omni_dim=sd['omni_proj.2.weight'].shape[0] if 'omni_proj.2.weight' in sd else 0)
            model = EventCaptionTransformer(**mcfg).to(dev)
            model.load_state_dict(sd)
            model.eval()
        dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=a.workers, collate_fn=trivial_batch_collator)
        results = {}
        with torch.no_grad():
            for batch in dl:
                for x in batch:
                    x['feats'] = {k: v.to(dev) for k, v in x['feats'].items()}
                for r in model(batch):
                    results[r['video_id']] = [[float(st), float(en), float(c)]
                                              for (st, en), c in zip(r['segments'].tolist(), r['scores'].tolist())]
        out = os.path.join(PREDS, '%s_%s.json' % (name, s))
        with open(out, 'w') as f:
            json.dump({'model': '%s (%s)' % (name, os.path.relpath(a.ckpt, ROOT)), 'set': s,
                       'videos': sorted(results), 'results': results}, f)
        print('%s: %d videos -> %s' % (s, len(results), out), flush=True)


# ======================================= SCORE =======================================================
def gt_spans(name):
    """{video: (n, 2) distinct GT spans} of the set's subset."""
    with open(anno_file(name)) as f:
        db = json.load(f)['database']
    return {v: np.array(sorted({tuple(round(x, 3) for x in a['segment']) for a in d['annotations']}), dtype=np.float64)
            for v, d in db.items() if d['subset'].lower() == SETS[name][1]}


def dedup(preds):
    """Identical spans (to the ms) are one prediction with the highest score; best score first."""
    best = {}
    for p in preds:
        k = (round(p[0], 3), round(p[1], 3))
        best[k] = max(best.get(k, -1.0), p[2])
    return sorted(([s, e, c] for (s, e), c in best.items()), key=lambda p: -p[2])


def ious(s, e, g):
    inter = np.clip(np.minimum(e, g[:, 1]) - np.maximum(s, g[:, 0]), 0, None)
    return inter / np.maximum((e - s) + (g[:, 1] - g[:, 0]) - inter, 1e-9)


def average_precision(gt, preds, t):
    """ActivityNet detection AP at tIoU t: preds (video, start, end, score) pooled over videos, best score first;
    each prediction takes the unmatched GT event of its video with the highest IoU >= t."""
    npos = sum(len(g) for g in gt.values())
    used = {v: np.zeros(len(g), bool) for v, g in gt.items()}
    tp = np.zeros(len(preds))
    for i, (v, s, e, _) in enumerate(preds):
        g = gt[v]
        if not len(g):
            continue
        o = ious(s, e, g)
        for j in np.argsort(-o, kind='stable'):
            if o[j] < t:
                break
            if not used[v][j]:
                used[v][j] = True
                tp[i] = 1
                break
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / max(npos, 1), ctp / np.maximum(ctp + cfp, 1e-9)
    mprec, mrec = np.hstack([0, prec, 0]), np.hstack([0, rec, 1])
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def scores(gt, results, videos):
    """Metrics in % over the given videos; results {video: deduped predictions}."""
    gt = {v: gt[v] for v in videos}
    pooled = sorted(((v, s, e, c) for v in videos for s, e, c in results.get(v, [])), key=lambda p: -p[3])
    out = {'AP@%.1f' % t: 100 * average_precision(gt, pooled, t) for t in TIOUS}
    out['mAP'] = float(np.mean([out['AP@%.1f' % t] for t in TIOUS]))
    best = []
    for v, g in gt.items():
        top = np.array([p[:2] for p in results.get(v, [])[:len(g)]], dtype=np.float64).reshape(-1, 2)
        best += [max(ious(s, e, top), default=0.0) if len(top) else 0.0 for s, e in g]
    best = np.array(best)
    for t in (0.3, 0.5, 0.7):
        out['R@%.1f' % t] = 100 * float(np.mean(best >= t))
    out['mIoU'] = 100 * float(best.mean())
    out['videos'], out['events'] = len(gt), sum(len(g) for g in gt.values())
    return out


COLS = ['AP@0.1', 'AP@0.3', 'AP@0.5', 'AP@0.7', 'AP@0.9', 'mAP', 'R@0.3', 'R@0.5', 'R@0.7', 'mIoU']


def score(a):
    report = {}
    for s in a.sets:
        files = sorted(glob.glob(os.path.join(a.preds, '*_%s.json' % s)))
        if not files:
            print('%s: no predictions in %s' % (s, a.preds))
            continue
        gt = gt_spans(s)
        runs = []
        for f in files:
            with open(f) as fh:
                p = json.load(fh)
            unknown = set(p['videos']) - set(gt)
            assert not unknown, '%s: %d videos not in the %s annotations' % (f, len(unknown), s)
            runs.append((p['model'], set(p['videos']), {v: dedup(r) for v, r in p['results'].items()}))
        common = set.intersection(*(v for _, v, _ in runs))
        print('\n%s (%s): class-agnostic, tIoU 0.1:0.9 for mAP' % (s, SETS[s][1]))
        print('%-46s %-6s %5s %6s ' % ('model', 'videos', 'n', 'events') + ' '.join('%6s' % c for c in COLS))
        rows = []
        for model, vids, res in runs:
            for label, sub in (('common', common), ('all', vids)):
                if label == 'all' and vids == common:
                    continue
                m = scores(gt, res, sorted(sub))
                rows.append(dict(m, model=model, videos=label, n=len(sub)))
                print('%-46s %-6s %5d %6d ' % (model[:46], label, len(sub), m['events'])
                      + ' '.join('%6.2f' % m[c] for c in COLS))
        report[s] = rows
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1)
        print('\n-> %s' % a.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['predict', 'score'])
    p.add_argument('--sets', nargs='+', default=list(SETS), choices=list(SETS))
    p.add_argument('--ckpt', help='predict: our checkpoint (training or API)')
    p.add_argument('--run', help='predict: run record of a training checkpoint (experiments/runs/*.json)')
    p.add_argument('--name', help='predict: model name in the file names (default: the run)')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--preds', default=PREDS, help='score: folder of prediction files')
    p.add_argument('--out', help='score: also write every number to this json')
    a = p.parse_args()
    predict(a) if a.cmd == 'predict' else score(a)
