import pickle, sys, numpy as np
res = pickle.load(open(sys.argv[1], 'rb'))
def iou_m(a, b):
    i = (np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    return i / np.maximum((a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None] - i, 1e-6)
for r in res: r['G'] = iou_m(r['pred'], r['gt'])
bi = np.concatenate([r['G'].max(1) for r in res])
for key in ('po_pt', 'po_q'):
    po = np.concatenate([r[key] for r in res])
    print('%s mean p(other): hit %.3f  boundary-off %.3f  wrong place %.3f' % (key, po[bi >= .5].mean(), po[(bi >= .3) & (bi < .5)].mean(), po[bi < .3].mean()))
def hits(idx, key, lam):
    h = n = 0
    for i in idx:
        r = res[i]; s = np.log(np.maximum(r['score'], 1e-8)) + lam * np.log(np.maximum(1 - r[key], 1e-6))
        top = np.argsort(-s)[:len(r['gt'])]; h += (r['G'][top].max(0) >= .5).sum(); n += len(r['gt'])
    return 100 * h / n
lams = [0, .25, .5, 1, 2, 4]
allidx = range(len(res))
for key in ('po_pt', 'po_q'):
    print(key, 'whole val:', '  '.join('%g:%.2f' % (l, hits(allidx, key, l)) for l in lams))
    g = []
    for seed in range(10):
        idx = np.random.RandomState(seed).permutation(len(res)); h = [idx[:len(idx)//2], idx[len(idx)//2:]]
        for a, b in ((0, 1), (1, 0)):
            l = max(lams, key=lambda l: hits(h[a], key, l)); g.append(hits(h[b], key, l) - hits(h[b], key, 0))
    print('   2-fold gain %.2f +- %.2f' % (np.mean(g), np.std(g)))
