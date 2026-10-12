import pickle, sys, itertools, numpy as np, torch
sys.path.insert(0, sys.argv[2]); from ens import segments
store = pickle.load(open(sys.argv[1], 'rb'))
def iou(p, g):
    i = max(0, min(p[1], g[1]) - max(p[0], g[0])); u = max(p[1], g[1]) - min(p[0], g[0])
    return i / u if u > 0 else 0
def evaluate(names, vote=None, per_video=False):
    rows = []
    for s in store:
        hs = [s['h'][n] for n in names]
        ev = torch.stack([h['ev'] for h in hs]).mean(0); qu = torch.stack([h['qu'] for h in hs]).mean(0)
        bd = torch.stack([h['bd'] for h in hs]).mean(0)
        P = segments(s['v'], ev, qu, bd, hs[0]['valid'], vote=vote)[:len(s['gt'])]
        b = [max([iou(p, g) for p in P], default=0) for g in s['gt']]
        rows.append([sum(x >= .5 for x in b), sum(x >= .7 for x in b), len(b)])
    rows = np.array(rows)
    return rows if per_video else (100 * rows[:, 0].sum() / rows[:, 2].sum(), 100 * rows[:, 1].sum() / rows[:, 2].sum())
names = list(store[0]['h'])
print('single models:')
for n in names: print('  %-9s R@0.5 %.2f  R@0.7 %.2f' % (n, *evaluate([n])))
print('pre-registered: all 5, no voting  R@0.5 %.2f  R@0.7 %.2f' % evaluate(names))
print('pre-registered: all 5, voting 0.75 R@0.5 %.2f  R@0.7 %.2f' % evaluate(names, vote=.75))
print('voting 0.75 on all_b16 alone     R@0.5 %.2f  R@0.7 %.2f' % evaluate(['all_b16'], vote=.75))
print('exploratory subsets:')
for k in (2, 3):
    for c in itertools.combinations(names, k):
        print('  %-30s R@0.5 %.2f  R@0.7 %.2f' % ('+'.join(c), *evaluate(list(c))))
