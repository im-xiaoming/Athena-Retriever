"""Readable side-by-side print of predicted events and the YouCook2 ground truth.

    import uniav_api as uv
    r = uv.describe_features(v, a, duration, video_id='-Ju39A-G0Dk')
    uv.show(r)                       # GT looked up from the YouCook2 annotations by video_id
    uv.show(r, gt=[(12.0, 30.5, 'cut the onion'), ...])   # or your own GT
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


def _rows(items, cap_width):
    """items: (index, time, extra cells..., caption, link). Captions wrap onto extra lines."""
    out = []
    for cells, caption, link in items:
        lines = textwrap.wrap(caption, cap_width) or ['']
        out.append(cells + '  ' + lines[0].ljust(cap_width) + '  ' + link)
        for more in lines[1:]:
            out.append(' ' * len(cells) + '  ' + more)
    return out


def format_events(result, gt='auto', iou_thr=0.5, cap_width=46):
    """Text block: predicted events, ground truth, and how they match. Returns a string."""
    if gt == 'auto':
        gt = load_gt(result.get('video_id', ''))
    preds = [(e['start'], e['end'], e['caption'], e.get('score', 0.0)) for e in result['events']]
    lines = ['=' * 100,
             'Video %s   duration %s   predicted %d   ground truth %d'
             % (result.get('video_id', '?'), _t(result.get('duration', 0)), len(preds), len(gt)),
             '=' * 100]

    best_gt = [max(((_iou(p[:2], g[:2]), j) for j, g in enumerate(gt)), default=(0.0, -1)) for p in preds]
    best_pred = [max(((_iou(p[:2], g[:2]), i) for i, p in enumerate(preds)), default=(0.0, -1)) for g in gt]

    lines.append('PREDICTED')
    lines.append('  #   time            score  %s  best GT (IoU)' % 'caption'.ljust(cap_width))
    items = []
    for i, (p, (v, j)) in enumerate(zip(preds, best_gt)):
        link = ('GT%-2d (%.2f)%s' % (j + 1, v, ' ok' if v >= iou_thr else '')) if j >= 0 else '-'
        items.append(('  %-3d %5s - %-6s  %.2f ' % (i + 1, _t(p[0]), _t(p[1]), p[3]), p[2], link))
    lines += _rows(items, cap_width)

    if gt:
        lines.append('')
        lines.append('GROUND TRUTH')
        lines.append('  #   time                   %s  best pred (IoU)' % 'caption'.ljust(cap_width))
        items = []
        for j, (g, (v, i)) in enumerate(zip(gt, best_pred)):
            link = ('P%-3d (%.2f)%s' % (i + 1, v, ' ok' if v >= iou_thr else '')) if i >= 0 else '-'
            items.append(('  GT%-2d %5s - %-6s      ' % (j + 1, _t(g[0]), _t(g[1])), g[2], link))
        lines += _rows(items, cap_width)

        tp_pred = sum(v >= iou_thr for v, _ in best_gt)
        tp_gt = sum(v >= iou_thr for v, _ in best_pred)
        lines.append('')
        lines.append('MATCH @ IoU %.1f: %d / %d predictions hit a GT event (precision %.2f), '
                     '%d / %d GT events are found (recall %.2f)'
                     % (iou_thr, tp_pred, len(preds), tp_pred / max(len(preds), 1),
                        tp_gt, len(gt), tp_gt / max(len(gt), 1)))
    return '\n'.join(lines)


def show(result, gt='auto', iou_thr=0.5, cap_width=46):
    print(format_events(result, gt=gt, iou_thr=iou_thr, cap_width=cap_width))
