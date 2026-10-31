"""HT-Step in the file formats the dataset class (libs/datasets/htstep.py) reads.

  python tools/htstep_data.py build [--upload]   # PC, uniav-api-env, GPU: text vectors of the steps
  python tools/htstep_data.py fetch              # training machine: official files, step vectors, features

The annotations and the taxonomy are the official ones (github.com/facebookresearch/htstep, cloned into
data/htstep by fetch): data/htstep/data/annotations.json and data/htstep/data/taxonomy.csv.

build writes data/htstep/step_emb_iv2j.npz: ids (global_step_index), emb (4958, 512) InternVideo2 text vectors
of "<activity>: <headline>" centred on their mean, text, mean. --upload puts it in the repo
(text_feats/step_emb_iv2j.npz).

fetch puts the k400_f16 features (k768 + a768) of the annotated videos in data/htstep/k400_feats, from the
feats/*.tar of nguyenminh04/htstep-feats-k400 (train) and nguyenminh04/htstep-val-seen-feats (val_seen).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import tarfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = 'nguyenminh04/htstep-text-feats'     # step vectors
FEATS_REPOS = ('nguyenminh04/htstep-feats-k400', 'nguyenminh04/htstep-val-seen-feats')  # k400_f16 features
OUT = os.path.join(ROOT, 'data', 'htstep')
OFFICIAL = 'https://github.com/facebookresearch/htstep'


def token():
    tok = os.environ.get('HF_TOKEN')
    if not tok and os.path.exists(os.path.join(ROOT, '.env')):
        for line in open(os.path.join(ROOT, '.env')):
            if line.startswith('HF_TOKEN='):
                tok = line.strip().split('=', 1)[1]
    return tok


def build(a):
    import torch
    from iv2_text import InternVideo2TextEncoder
    with open(os.path.join(OUT, 'data', 'taxonomy.csv'), newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r['global_step_index']))
    # the activity in front: 4958 steps but 4594 distinct headlines (e.g. "Add salt." in several activities)
    texts = ['%s: %s' % (r['activity'], r['headline'].strip()) for r in rows]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = InternVideo2TextEncoder(os.path.join(ROOT, 'ckpt', 'internvideo2', 'InternVideo2-stage2_1b-224p-f4.pt'),
                                  dev, torch.float32)
    raw = np.concatenate([enc(texts[i:i + 256]).cpu().numpy() for i in range(0, len(texts), 256)])
    mean = raw.mean(0)
    emb = raw - mean
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out = os.path.join(OUT, 'step_emb_iv2j.npz')
    np.savez_compressed(out, ids=np.array([int(r['global_step_index']) for r in rows]),
                        emb=emb.astype(np.float16), text=np.array(texts, dtype=str), mean=mean)
    print('htstep: %d steps; mean pairwise cosine %.2f -> %s' % (
        len(texts), (emb @ emb.T)[np.triu_indices(len(emb), 1)].mean(), out), flush=True)
    if a.upload:
        from huggingface_hub import HfApi
        HfApi(token=token()).upload_file(path_or_fileobj=out, path_in_repo='text_feats/step_emb_iv2j.npz',
                                         repo_id=REPO, repo_type='dataset',
                                         commit_message='text_feats/step_emb_iv2j.npz (tools/htstep_data.py build)')
        print('uploaded to %s' % REPO, flush=True)


def fetch(a):
    from huggingface_hub import hf_hub_download, snapshot_download
    if not os.path.exists(os.path.join(OUT, 'data', 'annotations.json')):
        subprocess.run(['git', 'clone', '--depth', '1', OFFICIAL, OUT], check=True)
    shutil.copy(hf_hub_download(REPO, 'text_feats/step_emb_iv2j.npz', repo_type='dataset', token=token()),
                os.path.join(OUT, 'step_emb_iv2j.npz'))
    with open(os.path.join(OUT, 'data', 'annotations.json')) as f:
        wanted = set(json.load(f))

    feats = os.path.join(OUT, 'k400_feats')
    os.makedirs(feats, exist_ok=True)
    n = 0
    for repo in FEATS_REPOS:
        d = snapshot_download(repo, repo_type='dataset', token=token(), allow_patterns=['feats/*.tar'],
                              local_dir=os.path.join(a.cache, repo.split('/')[1]) if a.cache else None)
        for t in sorted(os.listdir(os.path.join(d, 'feats'))):
            with tarfile.open(os.path.join(d, 'feats', t)) as tf:
                # only the annotated videos; empty files are failed extractions
                members = [m for m in tf.getmembers() if m.name.endswith('.npz') and m.size > 0
                           and os.path.basename(m.name)[:-4] in wanted]
                for m in members:
                    m.name = os.path.basename(m.name)
                tf.extractall(feats, members=members, filter='data')
                n += len(members)
    print(f'htstep: {n} of the {len(wanted)} annotated videos have features -> {feats}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['build', 'fetch'])
    p.add_argument('--upload', action='store_true')
    p.add_argument('--cache', help='fetch: download folder (default: the HF cache)')
    a = p.parse_args()
    build(a) if a.cmd == 'build' else fetch(a)
