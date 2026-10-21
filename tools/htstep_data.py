"""HT-Step in the file formats the dataset class (libs/datasets/step_captions.py) reads.

  python tools/htstep_data.py build [--upload]   # PC, uniav-api-env, GPU: annotations + caption vectors
  python tools/htstep_data.py fetch              # training machine: features, annotations, caption vectors

build writes, from anno/htstep.json of the features repo (tools/htstep_hub.py plan):
  data/htstep/annotations.json   {"database": {id: {"subset": "training" | "validation", "duration",
                                  "annotations": [{"segment": [s, e], "sentence": str}, ...]}}}
  data/htstep/caption_emb_iv2j.npz  keys "<video>#<i>", emb, sentences, mean: InternVideo2 text vectors
                                  centred on the mean of the unique training captions
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
REPO = 'nguyenminh04/htstep-feats'
OUT = os.path.join(ROOT, 'data', 'htstep')
VAL_ROLES = {'val', 'val_seen'}


def token():
    tok = os.environ.get('HF_TOKEN')
    if not tok and os.path.exists(os.path.join(ROOT, '.env')):
        for line in open(os.path.join(ROOT, '.env')):
            if line.startswith('HF_TOKEN='):
                tok = line.strip().split('=', 1)[1]
    return tok


def raw_anno():
    """{id: {role, duration, segments, captions}} as tools/htstep_hub.py plan wrote it."""
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(REPO, 'anno/htstep.json', repo_type='dataset', token=token())) as f:
        return json.load(f)


def build(a):
    import torch
    sys.path.insert(0, ROOT)
    from athena.encoders.internvideo2_text import InternVideo2TextEncoder
    os.makedirs(OUT, exist_ok=True)
    db = {}
    for vid, v in sorted(raw_anno().items()):
        anns = []
        for (s, e), c in zip(v['segments'], v['captions']):
            s, e = max(0.0, float(s)), min(float(e), float(v['duration']))
            if e > s and c.strip():
                anns.append({'segment': [s, e], 'sentence': c.strip()})
        if anns:
            subset = 'validation' if v['role'] in VAL_ROLES else 'training'
            db[vid] = {'subset': subset, 'duration': v['duration'], 'annotations': anns}
    with open(os.path.join(OUT, 'annotations.json'), 'w') as f:
        json.dump({'database': db}, f)

    keys = ['%s#%d' % (vid, i) for vid in db for i in range(len(db[vid]['annotations']))]
    caps = [a_['sentence'] for vid in db for a_ in db[vid]['annotations']]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = InternVideo2TextEncoder(os.path.join(ROOT, 'ckpt', 'internvideo2', 'InternVideo2-stage2_1b-224p-f4.pt'),
                                  dev, torch.float32)
    uniq = sorted(set(caps))
    row = {c: i for i, c in enumerate(uniq)}
    raw = np.concatenate([enc(uniq[i:i + 256]).cpu().numpy() for i in range(0, len(uniq), 256)])
    train = sorted({row[x['sentence']] for v in db.values() if v['subset'] == 'training' for x in v['annotations']})
    mean = raw[train].mean(0)
    emb = raw - mean
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out = os.path.join(OUT, 'caption_emb_iv2j.npz')
    np.savez_compressed(out, keys=np.array(keys), emb=emb[[row[c] for c in caps]].astype(np.float16),
                        sentences=np.array(caps, dtype=str), mean=mean)
    e = emb[np.random.default_rng(0).permutation(train)[:3000]]
    print('htstep: %d train / %d val videos, %d captions (%d unique); pairwise cosine of train captions %.2f -> %s' % (
        sum(v['subset'] == 'training' for v in db.values()), sum(v['subset'] == 'validation' for v in db.values()),
        len(caps), len(uniq), (e @ e.T)[np.triu_indices(len(e), 1)].mean(), OUT), flush=True)
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=token())
        for local, remote in (('annotations.json', 'anno/annotations.json'), ('caption_emb_iv2j.npz', 'text_feats/caption_emb_iv2j.npz')):
            api.upload_file(path_or_fileobj=os.path.join(OUT, local), path_in_repo=remote, repo_id=REPO,
                            repo_type='dataset', commit_message='%s (tools/htstep_data.py build)' % remote)
        print('uploaded to %s' % REPO, flush=True)


def fetch(a):
    from huggingface_hub import snapshot_download
    feats = os.path.join(OUT, 'iv2_feats')
    os.makedirs(feats, exist_ok=True)
    d = snapshot_download(REPO, repo_type='dataset', token=token(), local_dir=a.cache or None,
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
        shutil.copy(os.path.join(d, src), os.path.join(OUT, dst))
    print('htstep: %d videos -> %s' % (n, feats), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['build', 'fetch'])
    p.add_argument('--upload', action='store_true')
    p.add_argument('--cache', help='fetch: download folder (default: the HF cache)')
    a = p.parse_args()
    build(a) if a.cmd == 'build' else fetch(a)
