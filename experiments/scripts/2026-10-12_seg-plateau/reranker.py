"""Proposal re-ranker: a small transformer over the top-100 proposals of one video predicts each proposal's
best IoU with a GT step. Attention between proposals gets a learned bias from their overlap (IoU, containment,
distance), so the model can learn when two overlapping proposals are duplicates and when they are two steps.

  python reranker.py <diag dir> A      train on YouCook2 train proposals, score val
  python reranker.py <diag dir> B      2-fold on val (train on one half, score the other), 3 splits
"""
import pickle, sys, math, numpy as np, torch
from torch import nn
from torch.nn import functional as F
torch.manual_seed(0); np.random.seed(0)
dev = torch.device('cuda')
N = 100

def iou_mat(a, b):
    i = (np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    u = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None] - i
    return i / np.maximum(u, 1e-6), i

def one_to_one(giou):
    y = np.zeros(len(giou), np.float32)
    if giou.size:
        best = giou.argmax(0)                     # best proposal of each GT step
        for g, i in enumerate(best):
            if giou[i, g] >= .5:
                y[i] = max(y[i], giou[i, g])
    return y

def prep(r):
    p, s, d = r['pred'][:N].astype(np.float32), r['score'][:N].astype(np.float32), r['dur']
    n = len(p); L = np.maximum(p[:, 1] - p[:, 0], 1e-3)
    order = np.argsort(np.argsort(-s))
    sc = np.stack([np.log(np.maximum(s, 1e-6)), order / N, p[:, 0] / d, p[:, 1] / d, (p[:, 0] + p[:, 1]) / 2 / d,
                   np.log(L / d), np.log(L), np.full(n, np.log(d)), s / max(s.max(), 1e-6)], -1).astype(np.float32)
    piou, inter = iou_mat(p, p)
    cdist = ((p[:, None, 0] + p[:, None, 1]) - (p[None, :, 0] + p[None, :, 1])) / 2 / d
    pair = np.stack([piou, inter / L[:, None], inter / L[None, :], cdist, np.log(L[:, None] / L[None, :])], -1).astype(np.float32)
    giou, _ = iou_mat(p, r['gt']) if len(r['gt']) else (np.zeros((n, 1)), None)
    return {'sc': sc, 'pair': pair, 'q': r['q'][:N].astype(np.float32), 'qpt': r['qpt'][:N].astype(np.float32),
            'y': one_to_one(giou), 'giou': giou, 'k': len(r['gt'])}

def collate(items):
    B = len(items); n = max(len(x['y']) for x in items)
    out = {'sc': torch.zeros(B, n, 9), 'pair': torch.zeros(B, n, n, 5), 'q': torch.zeros(B, n, 512),
           'qpt': torch.zeros(B, n, 512), 'y': torch.zeros(B, n), 'm': torch.zeros(B, n, dtype=torch.bool)}
    for b, x in enumerate(items):
        k = len(x['y'])
        for key in ('sc', 'q', 'qpt', 'y'):
            out[key][b, :k] = torch.from_numpy(x[key])
        out['pair'][b, :k, :k] = torch.from_numpy(x['pair']); out['m'][b, :k] = True
    return {k: v.to(dev) for k, v in out.items()}

class Layer(nn.Module):
    def __init__(self, d, h, p=0.1):
        super().__init__()
        self.h = h; self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p)
    def forward(self, x, bias, m):
        B, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = q @ k.transpose(-1, -2) / math.sqrt(d // self.h) + bias          # (B, h, n, n)
        a = a.masked_fill(~m[:, None, None, :], -1e4).softmax(-1)
        x = x + self.drop(self.o((a @ v).transpose(1, 2).reshape(B, n, d)))
        return x + self.drop(self.ff(self.n2(x)))

class Reranker(nn.Module):
    def __init__(self, d=256, h=4, layers=3):
        super().__init__()
        self.inp_q, self.inp_p = nn.Linear(512, d), nn.Linear(512, d)
        self.inp_s = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d))
        self.pair = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, h * layers))
        self.layers = nn.ModuleList([Layer(d, h) for _ in range(layers)]); self.h = h
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)
    def forward(self, b):
        x = self.inp_s(b['sc'])
        if USE_Q:
            x = x + self.inp_q(F.normalize(b['q'], dim=-1)) + self.inp_p(F.normalize(b['qpt'], dim=-1))
        bias = self.pair(b['pair']).permute(0, 3, 1, 2)                       # (B, h*L, n, n)
        for i, l in enumerate(self.layers):
            x = l(x, bias[:, i * self.h:(i + 1) * self.h], b['m'])
        return b['sc'][..., 0] + self.out(x).squeeze(-1)    # base log score + learned correction

def r_at(items, scores, t=.5):
    h = n = 0
    for x, s in zip(items, scores):
        top = np.argsort(-s)[:x['k']]
        b = x['giou'][top].max(0) if len(top) else np.zeros(x['k'])
        h += (b >= t).sum(); n += x['k']
    return 100 * h / n

def predict(model, items):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(items), 32):
            b = collate(items[i:i + 32]); lo = model(b).cpu().numpy()
            out += [lo[j, :len(x['y'])] for j, x in enumerate(items[i:i + 32])]
    return out

def fit(tr, va, epochs=None, log=False):
    epochs = epochs or EPOCHS
    model = Reranker().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    steps = epochs * math.ceil(len(tr) / 16); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(tr))
        for i in range(0, len(tr), 16):
            b = collate([tr[j] for j in perm[i:i + 16]])
            lo = model(b)
            if LOSS == 'bce':
                loss = F.binary_cross_entropy_with_logits(lo[b['m']], b['y'][b['m']])
            else:
                t = (b['y'] >= .5).float() * b['y'] * b['m']; ok = t.sum(1) > 0
                t = t[ok] / t[ok].sum(1, keepdim=True)
                loss = -(t * lo[ok].masked_fill(~b['m'][ok], -1e4).log_softmax(-1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if log and (ep + 1) % 10 == 0:
            print('  epoch %d loss %.4f val R@0.5 %.2f' % (ep + 1, loss.item(), r_at(va, predict(model, va))), flush=True)
    return model

D, mode = sys.argv[1], sys.argv[2]
LOSS = sys.argv[3] if len(sys.argv) > 3 else 'list'
import os
USE_Q = os.environ.get('USE_Q', '1') == '1'; EPOCHS = int(os.environ.get('EPOCHS', '40'))
val = [prep(r) for r in pickle.load(open(D + '/props_val.pkl', 'rb'))]
base = lambda items: r_at(items, [x['sc'][:, 0] for x in items])
print('check: ranking by the one-to-one target gives R@0.5 %.2f' % r_at(val, [x['y'] for x in val]))
def dups(items, scores):
    d = n = 0
    for x, s in zip(items, scores):
        top = np.argsort(-s)[:x['k']]; g = x['giou'][top]
        hit = g.max(1) >= .5; d += hit.sum() - len(set(g[hit].argmax(1))); n += len(top)
    return 100 * d / n
if mode == 'A':
    tr = [prep(r) for r in pickle.load(open(D + '/props_train.pkl', 'rb'))]
    print('base val R@0.5 %.2f R@0.7 %.2f' % (base(val), r_at(val, [x['sc'][:, 0] for x in val], .7)))
    m = fit(tr, val, log=True); p = predict(m, val)
    print('A: reranked val R@0.5 %.2f R@0.7 %.2f' % (r_at(val, p), r_at(val, p, .7)))
    torch.save(m.state_dict(), D + '/reranker_A.pt')
else:
    g = []
    for seed in range(3):
        idx = np.random.RandomState(seed).permutation(len(val)); h = [idx[:len(idx) // 2], idx[len(idx) // 2:]]
        for a, c in ((0, 1), (1, 0)):
            tr = [val[i] for i in h[a]]; te = [val[i] for i in h[c]]
            m = fit(tr, te); r = r_at(te, predict(m, te))
            g.append(r - base(te)); p = predict(m, te)
            print('  split %d: base %.2f -> %.2f (duplicate slots %.1f%% -> %.1f%%)' % (
                seed, base(te), r, dups(te, [x['sc'][:, 0] for x in te]), dups(te, p)), flush=True)
    print('B: gain %.2f +- %.2f' % (np.mean(g), np.std(g)))
