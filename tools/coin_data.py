"""COIN data for training: features from the HF repo and one compact annotation file with the seen/unseen split.

  python tools/coin_data.py fetch    # feats/coin_feats_*.tar of nguyenminh04/coin-feats -> data/coin/iv2_feats/<id>.npz
  python tools/coin_data.py anno     # data/coin/coin_anno.json (needs datasets/annotations/COIN.json + taxonomy.xlsx)

coin_anno.json holds, for every video with features (minus data/coin_exclude.txt):
  {"tasks": {task: {"domain", "seen"}}, "videos": {id: {"task", "subset", "role", "duration",
   "segments": [[s, e], ...], "labels": [str, ...]}}, "split": {...how it was made}}
role is "train" (seen task, COIN training subset), "test_seen" (seen task, COIN testing subset) or
"test_unseen" (unseen task, any subset: never used for training). In every domain (taxonomy.xlsx),
about UNSEEN_FRAC of the tasks are unseen, picked with a fixed seed, as OV-AVEBench holds out
21 of its 67 classes (docs/PLAN_openvocab.md).
"""
import argparse
import json
import os
import random
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = 'nguyenminh04/coin-feats'
OUT = os.path.join(ROOT, 'data', 'coin')
ANN = os.path.join(ROOT, 'datasets', 'annotations')
UNSEEN_FRAC, SEED = 0.3, 2026


def token():
    tok = os.environ.get('HF_TOKEN')
    if not tok and os.path.exists(os.path.join(ROOT, '.env')):
        for line in open(os.path.join(ROOT, '.env')):
            if line.startswith('HF_TOKEN='):
                tok = line.strip().split('=', 1)[1]
    return tok


def fetch(a):
    from huggingface_hub import snapshot_download
    feats = os.path.join(OUT, 'iv2_feats')
    os.makedirs(feats, exist_ok=True)
    d = snapshot_download(REPO, repo_type='dataset', allow_patterns=['feats/*.tar'], token=token(),
                          local_dir=a.cache or None)
    tars = sorted(os.listdir(os.path.join(d, 'feats')))
    n = 0
    for t in tars:
        with tarfile.open(os.path.join(d, 'feats', t)) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith('.npz')]
            tf.extractall(feats, members=members)
            n += len(members)
    print('%d videos from %d tars -> %s' % (n, len(tars), feats), flush=True)


def domains():
    """taxonomy.xlsx (sheet 1: Domains | Targets) -> {task: domain}, read without openpyxl."""
    z = zipfile.ZipFile(os.path.join(ANN, 'taxonomy.xlsx'))
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    strings = [''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall(ns + 'si')]
    out = {}
    for r in list(ET.fromstring(z.read('xl/worksheets/sheet1.xml')).iter(ns + 'row'))[1:]:
        cells = {}
        for c in r.findall(ns + 'c'):
            v = c.find(ns + 'v')
            if v is not None:
                cells[re.sub(r'\d', '', c.get('r'))] = strings[int(v.text)] if c.get('t') == 's' else v.text
        if 'A' in cells and 'B' in cells:
            out[cells['B']] = cells['A']
    return out


def anno(a):
    with open(os.path.join(ANN, 'COIN.json')) as f:
        db = json.load(f)['database']
    have = {f[:-4] for f in os.listdir(os.path.join(OUT, 'iv2_feats')) if f.endswith('.npz')}
    exclude = {l.strip() for l in open(os.path.join(ROOT, 'data', 'coin_exclude.txt'))
               if l.strip() and not l.startswith('#')}
    dom = domains()
    tasks = sorted({v['class'] for v in db.values()})
    assert set(tasks) <= set(dom), set(tasks) - set(dom)
    rng = random.Random(SEED)
    unseen = set()
    for d in sorted(set(dom.values())):
        ts = sorted(t for t in tasks if dom[t] == d)
        unseen |= set(rng.sample(ts, round(UNSEEN_FRAC * len(ts))))
    videos = {}
    for vid, v in sorted(db.items()):
        if vid not in have or vid in exclude:
            continue
        anns = sorted(v['annotation'], key=lambda x: x['segment'][0])
        role = 'test_unseen' if v['class'] in unseen else ('train' if v['subset'] == 'training' else 'test_seen')
        videos[vid] = {'task': v['class'], 'subset': v['subset'], 'role': role, 'duration': v['duration'],
                       'segments': [x['segment'] for x in anns], 'labels': [x['label'] for x in anns]}
    out = {'split': {'unseen_frac': UNSEEN_FRAC, 'seed': SEED, 'by': 'tasks sampled per taxonomy domain',
                     'excluded': sorted(exclude)},
           'tasks': {t: {'domain': dom[t], 'seen': t not in unseen} for t in tasks},
           'videos': videos}
    path = os.path.join(OUT, 'coin_anno.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    roles = {r: sum(1 for v in videos.values() if v['role'] == r) for r in ('train', 'test_seen', 'test_unseen')}
    lab = lambda r: len({l for v in videos.values() if v['role'] == r for l in v['labels']})
    seen_labels = {l for v in videos.values() if v['role'] != 'test_unseen' for l in v['labels']}
    new = {l for v in videos.values() if v['role'] == 'test_unseen' for l in v['labels']} - seen_labels
    print('tasks: %d seen, %d unseen | videos: %s | labels: train %d, test_unseen %d (%d never in a seen task)'
          % (len(tasks) - len(unseen), len(unseen), roles, lab('train'), lab('test_unseen'), len(new)))
    print('-> %s (%.1f MB)' % (path, os.path.getsize(path) / 1e6), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('fetch', 'anno'))
    p.add_argument('--cache', default='', help='fetch: where to keep the downloaded tars (default: HF cache)')
    a = p.parse_args()
    {'fetch': fetch, 'anno': anno}[a.cmd](a)
