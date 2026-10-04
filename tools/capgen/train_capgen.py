"""Caption generator on top of the event model: GPT-2 conditioned on a segment (ClipCap-style prefix).

  python tools/capgen/train_capgen.py --name prefix            # segment prefix only
  python tools/capgen/train_capgen.py --name rag --rag 5       # + the 5 best retrieved train captions

Input per segment (tools/capgen/dump_segments.py): q (512) and K = 8 level-0 feature tokens, each
mapped to a GPT-2 input embedding, followed (with --rag) by the text " candidates: c1 ; ... ; c5 ."
and " caption:", then the caption is generated. During training the true caption is removed from
the candidates, so the model cannot learn to copy it; at validation they are the model's own top 5.

Writes data/youcookii/capgen/<name>/: generated captions for val GT spans and matched predicted
segments, with CIDEr / METEOR / BLEU-4 next to the retrieval pick (MBR) on the same segments.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data', 'youcookii', 'capgen')


class Prefix(nn.Module):
    """q and the K feature tokens -> K + 1 GPT-2 input embeddings."""

    def __init__(self, d_in, d_model, n_tok):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.mlp = nn.Sequential(nn.Linear(d_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos = nn.Parameter(torch.zeros(n_tok, d_model))

    def forward(self, q, tok):
        x = torch.cat((q[:, None], tok), 1)                    # (B, K+1, 512)
        return self.mlp(self.norm(x)) + self.pos


def build_text(cands, n):
    return (' candidates: ' + ' ; '.join(cands[:n]) + ' .' if n else '') + ' caption:'


class Data:
    def __init__(self, z, idx, tokzr, rag, train):
        self.q = torch.from_numpy(z['q'][idx].astype(np.float32))
        self.tok = torch.from_numpy(z['tok'][idx].astype(np.float32))
        self.cap = [str(c).strip().lower() for c in z['cap'][idx]]
        pool = [str(p) for p in z['pool']]
        self.texts, self.targets = [], []
        for i, ci in enumerate(z['cand'][idx]):
            cands = [pool[j] for j in ci]
            if train:   # never show the true caption among the candidates
                cands = [c for c in cands if c.strip().lower() != self.cap[i]]
            self.texts.append(tokzr.encode(build_text(cands, rag)))
            self.targets.append(tokzr.encode(' ' + self.cap[i]) + [tokzr.eos_token_id])
        self.mbr = [pool[j] for j in z['mbr'][idx]]

    def __len__(self):
        return len(self.cap)


def embed_batch(gpt, prefix, data, ids, dev, with_target, pad_id):
    """Left-padded input embeddings [prefix, text, (target)] and labels (-100 outside the target)."""
    pre = prefix(data.q[ids].to(dev), data.tok[ids].to(dev))                    # (B, P, D)
    seqs = [data.texts[i] + (data.targets[i] if with_target else []) for i in ids]
    L = max(len(s) for s in seqs)
    tok = torch.full((len(ids), L), pad_id, dtype=torch.long)
    att = torch.zeros(len(ids), pre.shape[1] + L, dtype=torch.long)
    lab = torch.full((len(ids), pre.shape[1] + L), -100, dtype=torch.long)
    for b, (i, s) in enumerate(zip(ids, seqs)):
        tok[b, L - len(s):] = torch.tensor(s)
        att[b, :pre.shape[1]] = 1          # prefix always attended
        att[b, pre.shape[1] + L - len(s):] = 1
        if with_target:
            n_t = len(data.targets[i])
            lab[b, -n_t:] = torch.tensor(data.targets[i])
    emb = gpt.transformer.wte(tok.to(dev))
    # prefix first, then left padding, then text: move the padding before the prefix
    full = torch.cat((pre, emb), 1)
    order = []
    for b, s in enumerate(seqs):
        pad = L - len(s)
        order.append(list(range(pre.shape[1], pre.shape[1] + pad)) + list(range(pre.shape[1]))
                     + list(range(pre.shape[1] + pad, pre.shape[1] + L)))
    order = torch.tensor(order, device=dev)
    full = torch.gather(full, 1, order[..., None].expand(-1, -1, full.shape[-1]))
    att = torch.gather(att.to(dev), 1, order)
    lab = torch.gather(lab.to(dev), 1, order)
    return full, att, lab


def scores(gts, hyps):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    g = {i: [x] for i, x in enumerate(gts)}; h = {i: [x] for i, x in enumerate(hyps)}
    out = {'CIDEr': Cider().compute_score(g, h)[0] * 100, 'BLEU4': Bleu(4).compute_score(g, h, verbose=0)[0][3] * 100}
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        out['METEOR'] = Meteor().compute_score(g, h)[0] * 100
    except Exception as e:   # java can fail; keep the other scores
        print('METEOR failed:', e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--rag', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--beams', type=int, default=3)
    ap.add_argument('--gpt', default='openai-community/gpt2')
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(DATA, a.name); os.makedirs(out_dir, exist_ok=True)
    z = np.load(os.path.join(DATA, 'segments.npz'))
    tokzr = GPT2TokenizerFast.from_pretrained(a.gpt)
    gpt = GPT2LMHeadModel.from_pretrained(a.gpt).to(dev)
    split, kind = z['split'], z['kind']
    tr = Data(z, np.where(split == 'training')[0], tokzr, a.rag, True)
    vals = {k: Data(z, np.where((split == 'validation') & (kind == k))[0], tokzr, a.rag, False) for k in ('gt', 'pred')}
    prefix = Prefix(z['q'].shape[1], gpt.config.n_embd, z['tok'].shape[1] + 1).to(dev)
    params = [{'params': gpt.parameters(), 'lr': a.lr}, {'params': prefix.parameters(), 'lr': a.lr * 10}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 300) * max(0.0, 1 - s / steps))
    pad = tokzr.eos_token_id
    print('train %d segments, val gt %d / pred %d, rag %d, %d steps' % (len(tr), len(vals['gt']), len(vals['pred']), a.rag, steps), flush=True)
    for ep in range(a.epochs):
        gpt.train(); prefix.train(); t0 = time.time(); tot = 0
        perm = torch.randperm(len(tr)).tolist()
        for s in range(0, len(perm), a.bs):
            emb, att, lab = embed_batch(gpt, prefix, tr, perm[s:s + a.bs], dev, True, pad)
            pos = (att.cumsum(-1) - 1).clamp(min=0)   # left padding: positions as generate() computes them
            loss = gpt(inputs_embeds=emb, attention_mask=att, position_ids=pos, labels=lab).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(gpt.parameters()) + list(prefix.parameters()), 1.0)
            opt.step(); sch.step(); tot += loss.item()
        print('epoch %d  loss %.3f  %.0f s' % (ep + 1, tot / math.ceil(len(perm) / a.bs), time.time() - t0), flush=True)
    gpt.eval(); prefix.eval()
    result = {'name': a.name, 'rag': a.rag, 'epochs': a.epochs}
    for k, d in vals.items():
        hyps = []
        with torch.no_grad():
            for s in range(0, len(d), 64):
                ids = list(range(s, min(s + 64, len(d))))
                emb, att, _ = embed_batch(gpt, prefix, d, ids, dev, False, pad)
                gen = gpt.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=30, num_beams=a.beams,
                                   no_repeat_ngram_size=3, pad_token_id=pad, eos_token_id=pad)
                hyps += [tokzr.decode(g, skip_special_tokens=True).strip() for g in gen]
        result[k] = {'generated': scores(d.cap, hyps), 'retrieval_mbr': scores(d.cap, d.mbr)}
        json.dump([{'gt': g, 'generated': h, 'retrieved': r} for g, h, r in zip(d.cap, hyps, d.mbr)],
                  open(os.path.join(out_dir, 'val_%s.json' % k), 'w'), indent=1)
        print('%-4s generated %s | retrieval %s' % (k, {m: round(v, 2) for m, v in result[k]['generated'].items()},
                                                    {m: round(v, 2) for m, v in result[k]['retrieval_mbr'].items()}), flush=True)
    json.dump(result, open(os.path.join(out_dir, 'result.json'), 'w'), indent=1)
    torch.save({'prefix': prefix.state_dict(), 'gpt': gpt.state_dict(), 'args': vars(a)}, os.path.join(out_dir, 'model.pt'))


if __name__ == '__main__':
    main()
