import sys, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
exec(open(sys.argv[2] + '/ens_eval.py').read().split("names = list(store[0]['h'])")[0])
names = list(store[0]['h'])
ens = evaluate(names, per_video=True)
rng = np.random.RandomState(0); B = 2000
for ref in ('E1', 'all_b16'):
    one = evaluate([ref], per_video=True)
    d = []
    for _ in range(B):
        i = rng.randint(0, len(ens), len(ens))
        d.append(100 * ens[i, 0].sum() / ens[i, 2].sum() - 100 * one[i, 0].sum() / one[i, 2].sum())
    d = np.array(d)
    print('5-model ensemble - %-8s R@0.5 %+.2f, 95%% CI [%+.2f, %+.2f], P(gain <= 0) = %.3f' % (ref, d.mean(), *np.percentile(d, [2.5, 97.5]), (d <= 0).mean()))
