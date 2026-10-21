"""Score dumped predictions with the official densevid_eval tool and compare with libs/utils/seg_metrics.py.

  git clone https://github.com/ranjaykrishna/densevid_eval <dir>      # official tool, Python 2
  python train.py configs/youcook2_event.yaml --eval ckpt/<run>/best_seg.pth.tar --dump-preds <preds>
  python experiments/scripts/2026-10-13_std-metrics/official_dvc.py --tool <dir> --preds <preds> --thr 0.35

The official evaluate.py is Python 2. It is copied to a temporary folder and converted with lib2to3 (print
statements only) plus a `sets.Set = set` shim; its scoring code, evaluate_detection, is run unchanged. METEOR
is stubbed out: it is built in the constructor (a Java process) but only used for captions, not scored here.

Every video with no prediction left (score threshold) is left out of the submission: the tool would otherwise
reuse the previous video's last prediction index as its count. A left-out video scores precision 0 and
recall 0, as in seg_metrics.dvc_pr. COIN seen + unseen are pooled into one set (distinct videos), as in train.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from libs.utils.seg_metrics import TIOU_DVC, dvc_pr   # noqa: E402

SETS = {'youcook2': ['youcook2'], 'coin': ['unseen', 'seen'], 'anet': ['anet'], 'htstep': ['htstep']}


def load_official(tool):
    """The tool's ANETcaptions class, from a Python 3 copy of its evaluate.py."""
    work = tempfile.mkdtemp(prefix='densevid3_')
    shutil.copy(os.path.join(tool, 'evaluate.py'), work)
    with open(os.path.join(work, 'sets.py'), 'w') as f:
        f.write('Set = set\n')
    subprocess.run([sys.executable, '-m', 'lib2to3', '-w', '-n', '-f', 'print', os.path.join(work, 'evaluate.py')],
                   check=True, capture_output=True)
    sys.path.insert(0, work)
    import evaluate as ev
    ev.Meteor = lambda: None
    return ev.ANETcaptions, work


def official(cls, work, gt, pred, tious):
    """Average precision / recall over tious, as the tool's main() prints them."""
    gf, pf = os.path.join(work, 'gt.json'), os.path.join(work, 'pred.json')
    with open(gf, 'w') as f:
        json.dump(gt, f)
    with open(pf, 'w') as f:
        json.dump({'version': 'VERSION 1.0', 'external_data': {}, 'results': pred}, f)
    e = cls(ground_truth_filenames=[gf], prediction_filename=pf, tious=list(tious), max_proposals=1000)
    pr = [e.evaluate_detection(t) for t in tious]
    return float(np.mean([p for p, _ in pr])), float(np.mean([r for _, r in pr]))


def f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def main(a):
    cls, work = load_official(a.tool)
    print('%-9s %-10s %7s | %6s %6s %6s | %6s %6s %6s | %s' % (
        'set', 'cut', 'videos', 'P', 'R', 'F1', 'P', 'R', 'F1', 'max |diff|'))
    print('%-28s | %-20s | %s' % ('', 'official densevid_eval', 'seg_metrics.dvc_pr'))
    worst = 0.0
    for name, parts in SETS.items():
        gt, pred = {}, {}
        for part in parts:
            with open(os.path.join(a.preds, '%s_gt.json' % part)) as f:
                gt.update(json.load(f))
            with open(os.path.join(a.preds, '%s_pred.json' % part)) as f:
                pred.update(json.load(f))
        for cut, thr in (('all', None), ('thr %.2f' % a.thr, a.thr)):
            sub = {v: [p for p in ps if thr is None or p['score'] >= thr] for v, ps in pred.items()}
            sub = {v: ps for v, ps in sub.items() if ps}
            po, ro = official(cls, work, gt, sub, TIOU_DVC)
            videos = [(np.array([p['timestamp'] for p in pred[v]]).reshape(-1, 2),
                       np.array([p['score'] for p in pred[v]]), np.array(gt[v]['timestamps']).reshape(-1, 2))
                      for v in gt]
            pm, rm, fm = dvc_pr(videos, min_score=thr)
            d = max(abs(po - pm), abs(ro - rm))
            worst = max(worst, d)
            print('%-9s %-10s %7d | %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f | %.1e' % (
                name, cut, len(gt), 100 * po, 100 * ro, 100 * f1(po, ro), 100 * pm, 100 * rm, 100 * fm, d))
    print('largest difference: %.1e' % worst)
    shutil.rmtree(work)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tool', required=True, help='clone of github.com/ranjaykrishna/densevid_eval')
    p.add_argument('--preds', required=True, help='folder written by train.py --eval ... --dump-preds')
    p.add_argument('--thr', type=float, default=0.35, help='score threshold (chosen on the training videos)')
    main(p.parse_args())
