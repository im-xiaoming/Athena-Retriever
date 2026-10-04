"""ret_sim of generated captions: ONE-PEACE text cosine to the GT caption, next to the retrieval pick.

  python tools/capgen/score_retsim.py prefix rag5      # uses data/youcookii/capgen/<name>/val_*.json

The same ONE-PEACE text encoder (uniav_api, 6 GB checkpoint) embeds the GT, generated and
retrieved captions, so both columns are measured the same way. Results are added to result.json.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from uniav_api.config import Config, pick_device  # noqa: E402
from uniav_api.encoders.onepeace_text import TextEncoder  # noqa: E402

DATA = os.path.join(ROOT, 'data', 'youcookii', 'capgen')


def main():
    dev, dtype = pick_device('auto')
    enc = TextEncoder(Config.from_env().text_encoder, dev, dtype)   # honours UNIAV_TEXT_ENCODER
    cache = {}

    def emb(texts):
        new = sorted({t for t in texts if t not in cache})
        for i in range(0, len(new), 256):
            v = torch.nn.functional.normalize(enc(new[i:i + 256]), dim=-1).cpu().numpy()
            cache.update(zip(new[i:i + 256], v))
        return np.stack([cache[t] for t in texts])

    for name in sys.argv[1:]:
        res_path = os.path.join(DATA, name, 'result.json')
        res = json.load(open(res_path))
        for kind in ('gt', 'pred'):
            rows = json.load(open(os.path.join(DATA, name, 'val_%s.json' % kind)))
            g = emb([r['gt'] for r in rows])
            for col, key in (('generated', 'generated'), ('retrieval_mbr', 'retrieved')):
                v = emb([r[key] for r in rows])
                res[kind][col]['ret_sim'] = float((g * v).sum(1).mean())
            print('%-8s %-4s ret_sim generated %.4f | retrieval %.4f' % (
                name, kind, res[kind]['generated']['ret_sim'], res[kind]['retrieval_mbr']['ret_sim']), flush=True)
        json.dump(res, open(res_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
