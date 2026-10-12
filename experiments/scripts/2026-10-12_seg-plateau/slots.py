import pickle, sys, numpy as np
res = pickle.load(open(sys.argv[1], 'rb'))
def iou(p, g):
    i = max(0, min(p[1], g[1]) - max(p[0], g[0])); u = max(p[1], g[1]) - min(p[0], g[0])
    return i / u if u > 0 else 0
hit = dup = fp_near = fp_far = 0
for r in res:
    k = len(r['gt']); taken = set()
    for p in r['pred'][:k]:
        v = [iou(p, g) for g in r['gt']]; j = int(np.argmax(v))
        if v[j] >= .5:
            if j in taken: dup += 1
            else: hit += 1; taken.add(j)
        elif v[j] >= .3: fp_near += 1
        else: fp_far += 1
n = hit + dup + fp_near + fp_far
print('top-K slots: %.1f%% hit a new GT, %.1f%% duplicate of a GT already hit, %.1f%% IoU 0.3-0.5 (boundary off), %.1f%% IoU < 0.3 (wrong place / wrong granularity)' % tuple(100*x/n for x in (hit, dup, fp_near, fp_far)))
# the IoU < 0.3 slots: background, a merge of several GT steps, a piece of one step
bg = merge = part = other = 0
for r in res:
    k = len(r['gt'])
    for p in r['pred'][:k]:
        v = [iou(p, g) for g in r['gt']]
        if max(v) >= .3: continue
        inter = [max(0, min(p[1], g[1]) - max(p[0], g[0])) for g in r['gt']]
        L = p[1] - p[0]; cov = sum(inter) / max(L, 1e-6)
        n_touch = sum(i > 0.2 * (g[1]-g[0]) for i, g in zip(inter, r['gt']))
        if cov < .3: bg += 1
        elif n_touch >= 2: merge += 1
        elif L < 0.5 * min((g[1]-g[0]) for i, g in zip(inter, r['gt']) if i == max(inter)): part += 1
        else: other += 1
n = bg + merge + part + other
print('IoU<0.3 slots: %.0f%% mostly background (<30%% inside any step), %.0f%% span 2+ steps (merge), %.0f%% a small piece of one step, %.0f%% other' % tuple(100*x/n for x in (bg, merge, part, other)))
