"""Rerank the post-NMS segments by score * exp(lam * match); lam chosen on one half of val, scored on the other."""
import pickle, sys, numpy as np
res = pickle.load(open(sys.argv[1], 'rb'))
def iou(p, g):
    i = max(0, min(p[1], g[1]) - max(p[0], g[0])); u = max(p[1], g[1]) - min(p[0], g[0])
    return i / u if u > 0 else 0
for r in res:   # IoU of every prediction with every GT, once
    r['M'] = np.array([[iou(p, g) for g in r['gt']] for p in r['pred']]).reshape(len(r['pred']), len(r['gt']))
def metrics(vids, key, lam):
    c = {.3: 0, .5: 0, .7: 0}; n = 0; mi = []
    for r in vids:
        s = np.log(np.maximum(r['score'], 1e-8)) + lam * r['match'][key]
        k = len(r['gt']); top = np.argsort(-s)[:k]
        b = r['M'][top].max(0) if len(top) else np.zeros(k)
        for t in c: c[t] += (b >= t).sum()
        n += k; mi += list(b)
    return {**{'R@%.1f' % t: 100 * v / n for t, v in c.items()}, 'mIoU': 100 * np.mean(mi)}
# does the match separate the slots? (all 100 predictions: best IoU >= 0.5 vs < 0.3)
for key in ('max', 'top20', 'lse'):
    m = np.concatenate([r['match'][key] for r in res]); bi = np.concatenate([r['M'].max(1) for r in res])
    print('%-6s mean match: hit (IoU>=.5) %.3f  boundary-off %.3f  wrong place (<.3) %.3f' % (
          key, m[bi >= .5].mean(), m[(bi >= .3) & (bi < .5)].mean(), m[bi < .3].mean()))
lams = [0, 1, 2, 3, 5, 8, 12, 20, 30]
print('\nwhole val (for reference, lam picked on it would be optimistic):')
for key in ('max', 'top20', 'lse'):
    print(key, '  '.join('%g:%.2f' % (l, metrics(res, key, l)['R@0.5']) for l in lams))
# honest: 2-fold by video, several random splits
print('\n2-fold (lam picked on one half, R@0.5 on the other), 5 random splits:')
for key in ('max', 'top20', 'lse'):
    gains, picked = [], []
    for seed in range(5):
        idx = np.random.RandomState(seed).permutation(len(res)); halves = [idx[:len(idx)//2], idx[len(idx)//2:]]
        for a, b in ((0, 1), (1, 0)):
            A = [res[i] for i in halves[a]]; B = [res[i] for i in halves[b]]
            lb = max(lams, key=lambda l: metrics(A, key, l)['R@0.5'])
            picked.append(lb); gains.append(metrics(B, key, lb)['R@0.5'] - metrics(B, key, 0)['R@0.5'])
    print('%-6s gain %.2f +- %.2f  (lam picked %s)' % (key, np.mean(gains), np.std(gains), sorted(set(picked))))
