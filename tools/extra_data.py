"""ActivityNet Captions and HT-Step as more training videos, in the file formats of YouCook2, so the same
dataset class (libs/datasets/youcook2_cap.py) reads all three.

  python tools/extra_data.py build anet [--upload]   # PC, uniav-api-env, GPU: annotations + caption vectors
  python tools/extra_data.py fetch anet              # training machine: features, annotations, caption vectors

build writes, from anno/<name>.json of the features repo (tools/coin_hub.py plan):
  data/<name>/annotations.json   {"database": {id: {"subset": "training" | "validation", "duration",
                                  "annotations": [{"segment": [s, e], "sentence": str}, ...]}}}
  data/<name>/caption_emb_iv2j.npz  keys "<video>#<i>", emb, sentences, other, mean: InternVideo2 text vectors
                                  centred on the joint YouCook2 + COIN mean (data/youcookii/caption_emb_iv2j.npz),
                                  so every dataset shares one text space
Training videos that are held out anywhere are dropped: YouCook2 validation, COIN test_seen / test_unseen,
the benchmarks (benchmark/unav100_all.json, benchmark/dcase_all.json) and the other dataset's validation videos.
--upload puts both files in the features repo (anno/annotations.json, text_feats/caption_emb_iv2j.npz).
"""
import argparse
import json
import os
import shutil
import sys
import tarfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOS = {'anet': 'nguyenminh04/anet-feats', 'htstep': 'nguyenminh04/htstep-feats'}
VAL_ROLES = {'val', 'val_seen'}


def token():
    tok = os.environ.get('HF_TOKEN')
    if not tok and os.path.exists(os.path.join(ROOT, '.env')):
        for line in open(os.path.join(ROOT, '.env')):
            if line.startswith('HF_TOKEN='):
                tok = line.strip().split('=', 1)[1]
    return tok


def raw_anno(name):
    """{id: {role, duration, segments, captions}} as tools/coin_hub.py plan wrote it."""
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(REPOS[name], 'anno/%s.json' % name, repo_type='dataset', token=token())) as f:
        return json.load(f)


def held_out():
    """YouTube ids that no training set may contain."""
    with open(os.path.join(ROOT, 'data', 'youcookii', 'annotations', 'youcookii_annotations_trainval.json')) as f:
        out = {k for k, v in json.load(f)['database'].items() if v['subset'].lower() == 'validation'}
    with open(os.path.join(ROOT, 'data', 'coin', 'coin_anno.json')) as f:
        out |= {k for k, v in json.load(f)['videos'].items() if v['role'] != 'train'}
    for b in ('unav100_all.json', 'dcase_all.json'):
        with open(os.path.join(ROOT, 'benchmark', b)) as f:
            out |= {k[:11] for k in json.load(f)['database']}   # DCASE ids are "<youtube id>_<start>_<end>"
    for n in REPOS:
        out |= {k for k, v in raw_anno(n).items() if v['role'] in VAL_ROLES}
    return out


def build(a):
    import torch
    sys.path.insert(0, ROOT)
    from athena.encoders.internvideo2_text import InternVideo2TextEncoder
    out_dir = os.path.join(ROOT, 'data', a.name)
    os.makedirs(out_dir, exist_ok=True)
    banned = held_out()
    db, dropped = {}, []
    for vid, v in sorted(raw_anno(a.name).items()):
        val = v['role'] in VAL_ROLES
        if not val and vid in banned:
            dropped.append(vid)
            continue
        anns = []
        for (s, e), c in zip(v['segments'], v['captions']):
            s, e = max(0.0, float(s)), min(float(e), float(v['duration']))
            if e > s and c.strip():
                anns.append({'segment': [s, e], 'sentence': c.strip()})
        if anns:
            db[vid] = {'subset': 'validation' if val else 'training', 'duration': v['duration'], 'annotations': anns}
    with open(os.path.join(out_dir, 'annotations.json'), 'w') as f:
        json.dump({'database': db, 'dropped_held_out': dropped}, f)

    keys = ['%s#%d' % (vid, i) for vid in db for i in range(len(db[vid]['annotations']))]
    caps = [a_['sentence'] for vid in db for a_ in db[vid]['annotations']]
    j = np.load(os.path.join(ROOT, 'data', 'youcookii', 'caption_emb_iv2j.npz'))
    mean, other = j['mean'], j['other']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = InternVideo2TextEncoder(os.path.join(ROOT, 'ckpt', 'internvideo2', 'InternVideo2-stage2_1b-224p-f4.pt'),
                                  dev, torch.float32)
    uniq = sorted(set(caps))
    raw = np.concatenate([enc(uniq[i:i + 256]).cpu().numpy() for i in range(0, len(uniq), 256)])
    emb = raw - mean
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    row = {c: i for i, c in enumerate(uniq)}
    out = os.path.join(out_dir, 'caption_emb_iv2j.npz')
    np.savez_compressed(out, keys=np.array(keys), emb=emb[[row[c] for c in caps]].astype(np.float16),
                        sentences=np.array(caps, dtype=str), other=other, mean=mean)
    train = [row[c] for vid in db if db[vid]['subset'] == 'training' for c in {x['sentence'] for x in db[vid]['annotations']}]
    e = emb[np.random.default_rng(0).permutation(sorted(set(train)))[:3000]]
    print('%s: %d train / %d val videos, %d dropped (held out elsewhere), %d captions (%d unique); '
          'pairwise cosine of train captions %.2f -> %s' % (
              a.name, sum(v['subset'] == 'training' for v in db.values()), sum(v['subset'] == 'validation' for v in db.values()),
              len(dropped), len(caps), len(uniq), (e @ e.T)[np.triu_indices(len(e), 1)].mean(), out_dir), flush=True)
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=token())
        for local, remote in (('annotations.json', 'anno/annotations.json'), ('caption_emb_iv2j.npz', 'text_feats/caption_emb_iv2j.npz')):
            api.upload_file(path_or_fileobj=os.path.join(out_dir, local), path_in_repo=remote, repo_id=REPOS[a.name],
                            repo_type='dataset', commit_message='%s (tools/extra_data.py build)' % remote)
        print('uploaded to %s' % REPOS[a.name], flush=True)


def fetch(a):
    from huggingface_hub import snapshot_download
    out_dir = os.path.join(ROOT, 'data', a.name)
    feats = os.path.join(out_dir, 'iv2_feats')
    os.makedirs(feats, exist_ok=True)
    d = snapshot_download(REPOS[a.name], repo_type='dataset', token=token(), local_dir=a.cache or None,
                          allow_patterns=['feats/*.tar', 'anno/annotations.json', 'text_feats/caption_emb_iv2j.npz'])
    n = 0
    for t in sorted(os.listdir(os.path.join(d, 'feats'))):
        with tarfile.open(os.path.join(d, 'feats', t)) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith('.npz') and m.size > 0]   # empty: failed extraction
            for m in members:
                m.name = os.path.basename(m.name)
            tf.extractall(feats, members=members)
            n += len(members)
    for src, dst in (('anno/annotations.json', 'annotations.json'), ('text_feats/caption_emb_iv2j.npz', 'caption_emb_iv2j.npz')):
        shutil.copy(os.path.join(d, src), os.path.join(out_dir, dst))
    print('%s: %d videos -> %s' % (a.name, n, feats), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['build', 'fetch'])
    p.add_argument('name', choices=sorted(REPOS))
    p.add_argument('--upload', action='store_true')
    p.add_argument('--cache', help='fetch: download folder (default: the HF cache)')
    a = p.parse_args()
    build(a) if a.cmd == 'build' else fetch(a)
