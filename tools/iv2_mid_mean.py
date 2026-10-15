"""Training-set mean of the InternVideo2 middle-layer features (iv2_mid/<id>.npz), subtracted by the dataset.

  python tools/iv2_mid_mean.py [--folder data/youcookii/iv2_mid] [--out data/youcookii/iv2_mid_mean.npz] [--upload]

Mean over every row of every YouCook2 training video, per key (m20, m30, m35). It is a fixed constant of the
model input, like an image mean, so it is computed once and kept on HF next to the features.
"""
import argparse
import json
import os

import numpy as np

ANNO = './data/youcookii/annotations/youcookii_annotations_trainval.json'
REPO = 'nguyenminh04/uniav-youcook2-data'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', default='./data/youcookii/iv2_mid')
    p.add_argument('--out', default='./data/youcookii/iv2_mid_mean.npz')
    p.add_argument('--upload', action='store_true', help='also put it on HF as iv2_mid_mean.npz')
    a = p.parse_args()
    db = json.load(open(ANNO))['database']
    ids = sorted(v for v, d in db.items() if d['subset'] == 'training'
                 and os.path.exists(os.path.join(a.folder, v + '.npz')))
    sums, n = {}, 0
    for v in ids:
        z = np.load(os.path.join(a.folder, v + '.npz'))
        for k in z.files:
            sums[k] = sums.get(k, 0) + z[k].astype(np.float64).sum(0)
        n += len(z[z.files[0]])
    np.savez(a.out, **{k: (s / n).astype(np.float32) for k, s in sums.items()})
    print('%d training videos, %d rows, keys %s -> %s' % (len(ids), n, sorted(sums), a.out))
    if a.upload:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=a.out, path_in_repo='iv2_mid_mean.npz', repo_id=REPO,
                            repo_type='dataset', commit_message='iv2_mid training-set mean')
