"""Compare predicted events with the YouCook2 ground truth, one row per pair, in time order.

    import uniav_api as uv
    r = uv.describe_features(v, a, duration, video_id='-Ju39A-G0Dk')
    uv.show(r)              # text: timeline bars + one table, GT step next to the prediction for it
    uv.compare_table(r)     # the same table as a pandas DataFrame (renders as HTML in notebooks)

Each real step is paired with at most one prediction (best overlap first). Verdicts:
  good     IoU >= 0.5          partial  0 < IoU < 0.5
  missed   no prediction overlaps this real step
  extra    a prediction that overlaps no real step
"""
import json
import os
import textwrap

from .config import ROOT

ANNOTATIONS = os.path.join(ROOT, 'data', 'youcookii', 'annotations', 'youcookii_annotations_trainval.json')
_db = None


def load_gt(video_id):
    """[(start, end, caption)] from the YouCook2 annotations, or [] if the video is unknown."""
    global _db
    if _db is None:
        with open(ANNOTATIONS) as f:
            _db = json.load(f)['database']
    x = _db.get(video_id)
    return [(a['segment'][0], a['segment'][1], a['sentence']) for a in x['annotations']] if x else []


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _t(sec):
    sec = int(round(sec))
    return '%d:%02d' % (sec // 60, sec % 60)


def pair_events(result, gt='auto'):
    """Rows in time order: {gt, pred, iou, verdict}; gt / pred are (start, end, caption) or None."""
    if gt == 'auto':
        gt = load_gt(result.get('video_id', ''))
    preds = [(e['start'], e['end'], e['caption']) for e in result['events']]
    pairs = sorted(((_iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, rows = set(), set(), []
    for v, i, j in pairs:   # greedy one-to-one matching, best overlap first
        if v <= 0 or i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        rows.append({'gt': gt[j], 'pred': preds[i], 'iou': v, 'verdict': 'good' if v >= 0.5 else 'partial'})
    rows += [{'gt': g, 'pred': None, 'iou': 0.0, 'verdict': 'missed'} for j, g in enumerate(gt) if j not in used_g]
    rows += [{'gt': None, 'pred': p, 'iou': 0.0, 'verdict': 'extra'} for i, p in enumerate(preds) if i not in used_p]
    rows.sort(key=lambda r: (r['gt'] or r['pred'])[0])
    return rows, gt, preds


def _bar(segs, duration, width):
    line = [' '] * width
    for s, e, _ in segs:
        a = int(s / max(duration, 1e-6) * width)
        b = max(a + 1, int(round(e / max(duration, 1e-6) * width)))
        for k in range(a, min(b, width)):
            line[k] = '#'
    return ''.join(line)


def format_events(result, gt='auto', width=96):
    rows, gt, preds = pair_events(result, gt)
    dur = float(result.get('duration') or max([s[1] for s in gt + preds] or [1]))
    n_good = sum(r['verdict'] == 'good' for r in rows)
    out = ['Video %s, %s long: the model found %d events, the real recipe has %d steps.'
           % (result.get('video_id', '?'), _t(dur), len(preds), len(gt)),
           'Matched well (IoU >= 0.5): %d of %d real steps.' % (n_good, len(gt)) if gt else '', '']
    bw = width - 8
    ticks = ''.join((_t(dur * k / 4)).ljust(bw // 4) for k in range(4)) + _t(dur)
    out += ['timeline  ' + ticks, 'real   |' + _bar(gt, dur, bw) + '|', 'model  |' + _bar(preds, dur, bw) + '|', '']

    cw = (width - 34) // 2
    head = '%-13s %-*s   %-13s %-*s  %s' % ('real time', cw, 'real step (ground truth)', 'model time', cw, 'model caption', 'verdict')
    out += [head, '-' * len(head)]
    for r in rows:
        g, p = r['gt'], r['pred']
        gl = textwrap.wrap(g[2], cw) if g else ['-']
        pl = textwrap.wrap(p[2], cw) if p else ['-']
        gt_time = '%s - %s' % (_t(g[0]), _t(g[1])) if g else ''
        pr_time = '%s - %s' % (_t(p[0]), _t(p[1])) if p else ''
        verdict = r['verdict'] + (' (IoU %.2f)' % r['iou'] if r['iou'] > 0 else '')
        for k in range(max(len(gl), len(pl))):
            out.append('%-13s %-*s   %-13s %-*s  %s' % (
                gt_time if k == 0 else '', cw, gl[k] if k < len(gl) else '',
                pr_time if k == 0 else '', cw, pl[k] if k < len(pl) else '', verdict if k == 0 else ''))
    return '\n'.join(out)


def show(result, gt='auto', width=110):
    print(format_events(result, gt=gt, width=width))


def compare_table(result, gt='auto'):
    """pandas DataFrame with one row per (real step, prediction) pair, in time order."""
    import pandas as pd
    rows, _, _ = pair_events(result, gt)
    return pd.DataFrame([{
        'real time': '%s - %s' % (_t(r['gt'][0]), _t(r['gt'][1])) if r['gt'] else '',
        'real step (ground truth)': r['gt'][2] if r['gt'] else '-',
        'model time': '%s - %s' % (_t(r['pred'][0]), _t(r['pred'][1])) if r['pred'] else '',
        'model caption': r['pred'][2] if r['pred'] else '-',
        'IoU': round(r['iou'], 2), 'verdict': r['verdict']} for r in rows])


def plot(result, gt='auto', ax=None, max_chars=38):
    """Timeline figure: real steps on top, the model's events below, captions on the bars.

    Bar colour follows the verdict of each pair (good / partial / missed / extra), as in show().
    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    rows, gt, preds = pair_events(result, gt)
    dur = float(result.get('duration') or max([s[1] for s in gt + preds] or [1]))
    colour = {'good': '#2e9e5b', 'partial': '#e0a030', 'missed': '#c0504d', 'extra': '#8c8c8c'}
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 1.2 + 0.42 * (len(gt) + len(preds))))
    lane = 0
    yt, yl = [], []
    for side, label in (('gt', 'real step'), ('pred', 'model')):
        for r in rows:
            seg = r[side]
            if seg is None:
                continue
            s, e, cap = seg
            ax.barh(lane, e - s, left=s, height=0.7, color=colour[r['verdict']], alpha=0.9)
            text = cap if len(cap) <= max_chars else cap[:max_chars - 1] + '…'
            ax.text(e + dur * 0.005, lane, text, va='center', fontsize=8.5)
            yt.append(lane); yl.append('%s  %s-%s' % (label, _t(s), _t(e)))
            lane += 1
        lane += 0.6
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, dur * 1.32)
    ax.set_xlabel('time (s)')
    n_good = sum(r['verdict'] == 'good' for r in rows)
    ax.set_title('%s: %d events found, %d real steps, %d matched with IoU >= 0.5'
                 % (result.get('video_id', '?'), len(preds), len(gt), n_good), fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in colour.items()], loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    return ax
