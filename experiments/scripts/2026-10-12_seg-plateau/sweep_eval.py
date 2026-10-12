import pickle, sys, numpy as np
allr = pickle.load(open(sys.argv[1], 'rb'))
def iou(p, g):
    i = max(0, min(p[1], g[1]) - max(p[0], g[0])); u = max(p[1], g[1]) - min(p[0], g[0])
    return i / u if u > 0 else 0
cfgs = []
for (ip, sig), res in allr.items():
    for r in res:
        r['M'] = np.array([[iou(p, g) for g in r['gt']] for p in r['pred']]).reshape(len(r['pred']), len(r['gt']))
        r['len'] = np.log(np.maximum((r['pred'][:, 1] - r['pred'][:, 0]) / r['dur'], 1e-4))
    for mu in (0, -0.1, -0.2, -0.3):     # length prior: log score + mu * log(relative length); mu < 0 favours shorter
        cfgs.append(((ip, sig, mu), res))
def per_video(res, mu):
    """hits at IoU .5/.7 and number of GT, per video"""
    out = []
    for r in res:
        s = np.log(np.maximum(r['score'], 1e-8)) + mu * r['len']
        k = len(r['gt']); b = r['M'][np.argsort(-s)[:k]].max(0)
        out.append(((b >= .5).sum(), (b >= .7).sum(), k))
    return np.array(out)
tab = {c: per_video(res, c[2]) for c, res in cfgs}
R = lambda c, idx, j=0: 100 * tab[c][idx, j].sum() / tab[c][idx, 2].sum()
base = (0.3, 0.5, 0); allidx = np.arange(len(tab[base]))
print('whole val R@0.5 / R@0.7 (optimistic if picked here):')
for c in sorted(tab, key=lambda c: -R(c, allidx))[:6] + [base]:
    print('  iou_power %.1f sigma %.1f len_mu %+.1f : %.2f / %.2f' % (*c, R(c, allidx), R(c, allidx, 1)))
gains, gains7, picked = [], [], []
for seed in range(10):
    idx = np.random.RandomState(seed).permutation(len(allidx)); h = [idx[:len(idx)//2], idx[len(idx)//2:]]
    for a, b in ((0, 1), (1, 0)):
        best = max(tab, key=lambda c: R(c, h[a]))
        picked.append(best); gains.append(R(best, h[b]) - R(base, h[b])); gains7.append(R(best, h[b], 1) - R(base, h[b], 1))
print('2-fold x10 splits: R@0.5 gain %.2f +- %.2f, R@0.7 gain %.2f +- %.2f' % (np.mean(gains), np.std(gains), np.mean(gains7), np.std(gains7)))
from collections import Counter; print('picked', Counter(picked).most_common(4))
