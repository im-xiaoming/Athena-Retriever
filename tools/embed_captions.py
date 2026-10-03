"""Build data/youcookii/caption_emb.npz: 1536-d ONE-PEACE vectors for every YouCook2 caption.

  python tools/embed_captions.py

Runs in uniav-env on CPU, about 50 minutes for 11594 captions. Progress is saved every
640 captions, so a rerun resumes. Only videos with both video and audio features are
used, keyed "<video>#<segment index>" like the youcook2_cap dataset.
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONEPEACE = os.path.join(ROOT, 'ONEPEACE_extract_embd_code')
sys.path.insert(0, ONEPEACE)
from onepeace_text import from_pretrained  # noqa: E402

DATA = os.path.join(ROOT, 'data', 'youcookii')
PART = os.path.join(DATA, '_caption_emb_partial.npy')
OUT = os.path.join(DATA, 'caption_emb.npz')


def main():
    with open(os.path.join(DATA, 'annotations', 'youcookii_annotations_trainval.json')) as f:
        db = json.load(f)['database']
    feats = os.listdir(os.path.join(DATA, 'av_features'))
    vis = {x.replace('_one_peace_video_finetune.npy', '') for x in feats if 'video_finetune' in x}
    aud = {x.replace('_one_peace_audio.npy', '') for x in feats if x.endswith('_one_peace_audio.npy')}
    keys, caps = [], []
    for vid in sorted(db):
        if vid in vis & aud:
            for i, a in enumerate(db[vid]['annotations']):
                keys.append('%s#%d' % (vid, i)); caps.append(a['sentence'])
    n = len(caps)

    emb = np.load(PART) if os.path.exists(PART) else np.zeros((n, 1536), np.float32)
    done = int((np.abs(emb).sum(1) > 0).sum())
    print('total %d | done %d | left %d' % (n, done, n - done), flush=True)
    if done < n:
        model = from_pretrained(os.path.join(ONEPEACE, 'models', 'one-peace-text.pt'),
                                device='cpu', dtype='float32')
        bs, t0 = 64, time.time()
        for i in range(done, n, bs):
            with torch.no_grad():
                emb[i:i + bs] = model.extract_text_features(model.process_text(caps[i:i + bs])).numpy()
            if ((i - done) // bs) % 10 == 0:
                np.save(PART, emb)
                d = min(i + bs, n)
                print('%d/%d  ~%.1f min left' % (d, n, (n - d) * (time.time() - t0) / max(d - done, 1) / 60),
                      flush=True)
        np.save(PART, emb)
    # float16 keeps the file small; vectors are normalised so the error is negligible
    np.savez_compressed(OUT, keys=np.array(keys), emb=emb.astype(np.float16),
                        sentences=np.array(caps, dtype=object))
    if os.path.exists(PART):
        os.remove(PART)
    print('done: %s | %.1f MB' % (emb.shape, os.path.getsize(OUT) / 1e6), flush=True)


if __name__ == '__main__':
    main()
