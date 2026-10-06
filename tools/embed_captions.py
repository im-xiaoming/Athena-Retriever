"""Build data/youcookii/caption_emb_iv2.npz: InternVideo2 text vectors (512-d) of every YouCook2 caption.

  python tools/embed_captions.py            # uniav-api-env, GPU: under a minute
  python tools/embed_captions.py --coin     # data/coin_label_emb_iv2.npz: the 749 COIN step labels
  python tools/embed_captions.py --joint    # *_iv2j.npz: YouCook2 captions, COIN labels and "other", one joint centring

Every caption of the videos with InternVideo2 features (data/youcookii/iv2_feats), keyed
"<video>#<segment index>". InternVideo2's text tower is aligned with its video tower, whose 512-d
projection is the v512 input feature. These vectors are the caption space of the model and the
space of the txt_sim metric.

Raw InternVideo2 text vectors are strongly anisotropic: any two captions have cosine ~0.95, so the
soft targets of the embedding loss (softmax of caption-caption cosine / 0.02) would spread over
~7000 captions. The stored vectors are therefore centred on the mean of the unique train captions
and re-normalised (pairwise cosine 0.38, soft targets over ~50 captions); `mean` is stored so a new
sentence (uniav_api search) is mapped the same way: normalize(encode(s) - mean).
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from uniav_api.encoders.internvideo2_text import InternVideo2TextEncoder  # noqa: E402

DATA = os.path.join(ROOT, 'data', 'youcookii')
CKPT = os.path.join(ROOT, 'ckpt', 'internvideo2', 'InternVideo2-stage2_1b-224p-f4.pt')


def coin():
    """COIN step labels (datasets/annotations/COIN.json) -> data/coin_label_emb_iv2.npz.

    emb is centred on the COIN label mean (centring on the YouCook2 mean leaves them at pairwise
    cosine 0.63; on their own mean 0.01); raw (normalised, uncentred), mean and mean_youcook2 are
    kept so a joint model can choose its own centring.
    """
    with open(os.path.join(ROOT, 'datasets', 'annotations', 'COIN.json')) as f:
        d = json.load(f)['database']
    labels = sorted({a['label'] for v in d.values() for a in v['annotation']})
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = InternVideo2TextEncoder(CKPT, dev, torch.float32)
    raw = np.concatenate([enc(labels[i:i + 256]).cpu().numpy() for i in range(0, len(labels), 256)])
    mean = raw.mean(0)
    emb = raw - mean
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out = os.path.join(ROOT, 'data', 'coin_label_emb_iv2.npz')
    np.savez_compressed(out, labels=np.array(labels, dtype=str), emb=emb.astype(np.float16), raw=raw.astype(np.float16),
                        mean=mean.astype(np.float32),
                        mean_youcook2=np.load(os.path.join(DATA, 'caption_emb_iv2.npz'))['mean'])
    print('done: %d labels -> %s' % (len(labels), out), flush=True)


def joint():
    """One text space for YouCook2 + COIN (docs/PLAN_openvocab.md): every vector centred on
    mean_joint = (mean of the unique YouCook2 train captions + mean of the COIN labels of training videos) / 2,
    then re-normalised. Centring on the YouCook2 mean alone leaves non-cooking sentences bunched together
    (COIN labels at pairwise cosine 0.63), which hurts sentences on any topic.

    -> data/youcookii/caption_emb_iv2j.npz (same keys as caption_emb_iv2.npz, plus other, mean)
    -> data/coin/label_emb_iv2j.npz (labels: every COIN label, emb, other, mean)
    "other" is the background text of OV-AVEL (its Table 3): steps outside every event are pulled to it.
    """
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = InternVideo2TextEncoder(CKPT, dev, torch.float32)
    encode = lambda xs: np.concatenate([enc(xs[i:i + 256]).cpu().numpy() for i in range(0, len(xs), 256)])
    keys, caps, first = youcook2_captions()
    yc = encode(caps)
    with open(os.path.join(ROOT, 'data', 'coin', 'coin_anno.json')) as f:
        videos = json.load(f)['videos']
    labels = sorted({l for v in videos.values() for l in v['labels']})
    train_labels = {l for v in videos.values() if v['role'] == 'train' for l in v['labels']}
    cl = encode(labels)
    other = encode(['other'])[0]
    mean = 0.5 * yc[list(first.values())].mean(0) + 0.5 * cl[[i for i, l in enumerate(labels) if l in train_labels]].mean(0)
    norm = lambda x: (x - mean) / np.linalg.norm(x - mean, axis=-1, keepdims=True)
    out1 = os.path.join(DATA, 'caption_emb_iv2j.npz')
    np.savez_compressed(out1, keys=keys, emb=norm(yc).astype(np.float16), sentences=np.array(caps, dtype=str),
                        other=norm(other).astype(np.float32), mean=mean.astype(np.float32))
    out2 = os.path.join(ROOT, 'data', 'coin', 'label_emb_iv2j.npz')
    np.savez_compressed(out2, labels=np.array(labels, dtype=str), emb=norm(cl).astype(np.float16),
                        other=norm(other).astype(np.float32), mean=mean.astype(np.float32))
    cos = lambda x: float((lambda e: (e @ e.T)[np.triu_indices(len(e), 1)].mean())(norm(x)))
    print('pairwise cosine: YouCook2 %.2f, COIN %.2f | done: %s, %s' % (
        cos(yc[list(first.values())][:3000]), cos(cl), out1, out2), flush=True)


def youcook2_captions():
    """Every caption of the YouCook2 videos with features: keys "<video>#<i>", sentences, and the index
    of the first occurrence of every unique train caption (the dataset's caption pool)."""
    with open(os.path.join(DATA, 'annotations', 'youcookii_annotations_trainval.json')) as f:
        db = json.load(f)['database']
    have = {f[:-4] for f in os.listdir(os.path.join(DATA, 'iv2_feats')) if f.endswith('.npz')}
    keys, caps = [], []
    for vid in sorted(db):
        if vid in have:
            for i, a in enumerate(db[vid]['annotations']):
                keys.append('%s#%d' % (vid, i)); caps.append(a['sentence'])
    first = {}
    for i, k in enumerate(keys):
        if db[k.split('#')[0]]['subset'] == 'training':
            first.setdefault(caps[i], i)
    return np.array(keys), caps, first


def main():
    if '--coin' in sys.argv:
        return coin()
    if '--joint' in sys.argv:
        return joint()
    keys, caps, first = youcook2_captions()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = InternVideo2TextEncoder(CKPT, dev, torch.float32)
    emb = np.concatenate([enc(caps[i:i + 256]).cpu().numpy() for i in range(0, len(caps), 256)])
    mean = emb[list(first.values())].mean(0)
    emb = emb - mean
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out = os.path.join(DATA, 'caption_emb_iv2.npz')
    np.savez_compressed(out, keys=keys, emb=emb.astype(np.float16), sentences=np.array(caps, dtype=str),
                        mean=mean.astype(np.float32))
    print('done: %s -> %s, %.1f MB' % (emb.shape, out, os.path.getsize(out) / 1e6), flush=True)


if __name__ == '__main__':
    main()
