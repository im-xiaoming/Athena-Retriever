import pickle, sys, numpy as np
res = pickle.load(open(sys.argv[1], 'rb'))
def iou(p, g):
    i = max(0, min(p[1], g[1]) - max(p[0], g[0])); u = max(p[1], g[1]) - min(p[0], g[0])
    return i / u if u > 0 else 0
def recall(kf, t=0.5):
    c = n = 0
    for r in res:
        k = kf(len(r['gt'])); P = r['pred'][:k]
        for g in r['gt']:
            n += 1; c += max([iou(p, g) for p in P], default=0) >= t
    return 100 * c / n
print('R@0.5 K=#GT %.2f | 2K %.2f | 3K %.2f | top100 %.2f' % (recall(lambda n: n), recall(lambda n: 2*n), recall(lambda n: 3*n), recall(lambda n: 100)))
print('R@0.3 K=#GT %.2f top100 %.2f | R@0.7 K=#GT %.2f top100 %.2f' % (recall(lambda n: n, .3), recall(lambda n: 100, .3), recall(lambda n: n, .7), recall(lambda n: 100, .7)))
# GT statistics
gl = np.concatenate([r['gt'][:, 1] - r['gt'][:, 0] for r in res]); dur = np.array([r['dur'] for r in res])
rel = np.concatenate([(r['gt'][:, 1] - r['gt'][:, 0]) / r['dur'] for r in res])
cov = np.array([(r['gt'][:, 1] - r['gt'][:, 0]).sum() / r['dur'] for r in res])
print('videos %d, GT %d, per video %.1f; video %.0f s; GT len median %.1f s, rel median %.3f; GT covers %.0f%% of a video' % (
      len(res), len(gl), len(gl)/len(res), dur.mean(), np.median(gl), np.median(rel), 100*cov.mean()))
# recall by GT length bucket (K = #GT)
rows = []
for r in res:
    P = r['pred'][:len(r['gt'])]
    for g in r['gt']:
        b = max([iou(p, g) for p in P], default=0)
        bt = max([iou(p, g) for p in r['pred']], default=0)
        rows.append(((g[1]-g[0]) / r['dur'], b, bt))
rows = np.array(rows)
for lo, hi in [(0, .02), (.02, .04), (.04, .08), (.08, .16), (.16, 1)]:
    m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    print('GT len %4.0f-%3.0f%% of video: %4d GT (%2.0f%%)  R@0.5 K=#GT %.1f  top100 %.1f' % (100*lo, 100*hi, m.sum(), 100*m.mean(), 100*(rows[m,1]>=.5).mean(), 100*(rows[m,2]>=.5).mean()))
# where do misses at K=#GT come from: nothing near (best over top100 < .5) vs ranked out
miss = rows[:, 1] < .5
print('misses %.1f%% of GT: %.1f%% no good proposal at all (top100 < 0.5), %.1f%% good proposal ranked below K' % (
      100*miss.mean(), 100*(miss & (rows[:,2] < .5)).mean(), 100*(miss & (rows[:,2] >= .5)).mean()))
# predicted vs GT lengths (top K)
pl = np.concatenate([(r['pred'][:len(r['gt']),1]-r['pred'][:len(r['gt']),0])/r['dur'] for r in res])
print('pred len rel median %.3f vs GT %.3f' % (np.median(pl), np.median(rel)))
