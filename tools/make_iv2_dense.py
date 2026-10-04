"""Interleave the default and the half-second shifted InternVideo2 features: 2 rows per second.

  python tools/make_iv2_dense.py      # data/youcookii/iv2_feats + iv2_feats_shift -> iv2_dense

Row i of the default extraction is centred on i + 0.5 s and row i of the shifted one on i + 1.0 s,
so the interleaved row j is centred on 0.5 j + 0.5 s: the grid of dataset.iv2_rows_per_sec = 2.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'data', 'youcookii')
KEYS = ('v768', 'v512', 'a768')


def main():
    src, sh, out = os.path.join(D, 'iv2_feats'), os.path.join(D, 'iv2_feats_shift'), os.path.join(D, 'iv2_dense')
    os.makedirs(out, exist_ok=True)
    done = missing = 0
    for f in sorted(os.listdir(src)):
        if not f.endswith('.npz'):
            continue
        if not os.path.exists(os.path.join(sh, f)):
            missing += 1; continue
        a, b = np.load(os.path.join(src, f)), np.load(os.path.join(sh, f))
        n = min(len(a['v768']), len(b['v768']))
        dense = {}
        for k in KEYS:
            x = np.empty((2 * n,) + a[k].shape[1:], a[k].dtype)
            x[0::2], x[1::2] = a[k][:n], b[k][:n]
            dense[k] = x
        np.savez(os.path.join(out, f), **dense)
        done += 1
    print('%d videos interleaved into %s, %d without shifted features' % (done, out, missing))


if __name__ == '__main__':
    main()
