"""Greedy pick of K segments in score order, skipping one that overlaps an already chosen segment by more than
`frac` of its own length (YouCook2 steps almost never overlap). 2-fold choice of frac on val halves."""
import pickle, sys, numpy as np
res = pickle.load(open(sys.argv[1], 'rb'))
def iou(p, g):
    i = max(0, min(p[1], g[1]) - max(p[0], g[0])); u = max(p[1], g[1]) - min(p[0], g[0])
    return i / u if u > 0 else 0
def pick(r, frac):
    k = len(r['gt']); chosen = []
    for i in np.argsort(-r['score']):
        p = r['pred'][i]; L = max(p[1] - p[0], 1e-6)
        if all(max(0, min(p[1], q[1]) - max(p[0], q[0])) <= frac * min(L, q[1] - q[0]) for q in chosen):
            chosen.append(p)
        if len(chosen) == k: break
    return chosen
def hits(vids, frac):
    h = n = 0
    for r in vids:
        P = pick(r, frac) if frac < 1 else list(r['pred'][np.argsort(-r['score'])[:len(r['gt'])]])
        for g in r['gt']:
            n += 1; h += max([iou(p, g) for p in P], default=0) >= .5
    return h, n
fr = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
tab = {f: [hits([r], f) for r in res] for f in fr}
R = lambda f, idx: 100 * sum(tab[f][i][0] for i in idx) / sum(tab[f][i][1] for i in idx)
allidx = range(len(res))
print('whole val R@0.5:', '  '.join('%.1f:%.2f' % (f, R(f, allidx)) for f in fr))
g = []
for seed in range(10):
    idx = np.random.RandomState(seed).permutation(len(res)); h = [idx[:len(idx)//2], idx[len(idx)//2:]]
    for a, b in ((0, 1), (1, 0)):
        best = max(fr, key=lambda f: R(f, h[a])); g.append(R(best, h[b]) - R(1.0, h[b]))
print('2-fold gain %.2f +- %.2f' % (np.mean(g), np.std(g)))
