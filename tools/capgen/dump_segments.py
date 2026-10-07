"""Segment representations of the event model, as input for a caption generator.

  python tools/capgen/dump_segments.py      # -> data/youcookii/capgen/segments.npz (uniav-api-env)

For every YouCook2 video with InternVideo2 features, the API checkpoint (ckpt/api/uniav_iv2.pth)
gives, per segment:
  q    (512)     the SegmentContext vector the retrieval captioner uses
  tok  (K, 512)  level-0 features of the Embed Head sampled at K points across the segment
  cand (10)      the 10 best train captions for q (pool indices), and mbr: the caption picked
Segments: train = GT steps + 2 copies with boundaries moved by up to 20% of the length (the
predicted boundaries are never exact); val = GT steps ('gt') and the model's own segments
matched to a GT step with IoU >= 0.3 ('pred', as in train.py: top #GT by score).
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import athena as uv  # noqa: E402

K = 8
OUT = os.path.join(ROOT, 'data', 'youcookii', 'capgen', 'segments.npz')


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def main():
    pipe = uv.load(index_dir=os.path.join(ROOT, 'athena', 'index'))
    m, spec, dev = pipe.model, pipe.spec, pipe.device
    db = json.load(open(os.path.join(ROOT, 'data', 'youcookii', 'annotations',
                                     'youcookii_annotations_trainval.json')))['database']
    feats = os.path.join(ROOT, 'data', 'youcookii', 'iv2_feats')
    have = set(spec.stored_ids(feats))
    rng = np.random.default_rng(0)
    rows = {k: [] for k in ('q', 'tok', 'cand', 'mbr', 'cap', 'vid', 'split', 'kind', 'seg')}
    for vid, x in sorted(db.items()):
        if vid not in have or x['subset'] not in ('training', 'validation'):
            continue
        fv, fa, n = spec.prepare(*spec.from_store(vid, feats))
        V, A = fv.to(dev)[None], fa.to(dev)[None]
        mask = torch.ones(1, 1, V.shape[-1], dtype=torch.bool, device=dev)
        fV, fA, msk = m.backbone(V, A, mask)
        _, raw0 = m.embed_head([torch.cat((v, a), 1) for v, a in zip(fV, fA)], msk)
        raw0, L = raw0[0], int(msk[0].sum())
        step = float((n - 1) * spec.stride + spec.window) / spec.max_seq_len   # frames per grid step

        def to_grid(sec):
            return np.asarray(sec, np.float32) * spec.fps / step - 0.5

        anns = x['annotations'][:16]
        gt_sec = [a['segment'] for a in anns]
        jobs = []   # (segment in seconds, caption, kind)
        if x['subset'] == 'training':
            for (s, e), a in zip(gt_sec, anns):
                jobs.append(((s, e), a['sentence'], 'gt'))
                for _ in range(2):
                    w = max(e - s, 1.0)
                    jobs.append(((s + rng.uniform(-0.2, 0.2) * w, e + rng.uniform(-0.2, 0.2) * w), a['sentence'], 'gtjit'))
        else:
            for (s, e), a in zip(gt_sec, anns):
                jobs.append(((s, e), a['sentence'], 'gt'))
            segs, scores, _ = m(V[0], A[0])
            order = np.argsort(-scores, kind='stable')[:len(gt_sec)]   # top #GT by score
            secs = spec.to_seconds(segs[order], n, x['duration'])
            for g, a in zip(gt_sec, anns):
                if len(secs):
                    j = int(np.argmax([iou(p, g) for p in secs]))
                    if iou(secs[j], g) >= 0.3:
                        jobs.append((tuple(secs[j]), a['sentence'], 'pred'))
        segs = torch.from_numpy(np.stack([to_grid(j[0]) for j in jobs])).to(dev)
        q = m.seg_ctx(raw0, L, segs)                                         # (N, 512), normalised
        t = torch.linspace(0, 1, K, device=dev)
        pos = (segs[:, :1] + t[None] * (segs[:, 1:] - segs[:, :1])).clamp(0, L - 1)  # (N, K) grid positions
        lo = pos.floor().long(); hi = (lo + 1).clamp(max=L - 1); w = (pos - lo.float())[..., None]
        r0 = raw0.t()                                                        # (T, 512)
        tok = r0[lo] * (1 - w) + r0[hi] * w                                 # (N, K, 512)
        s_all = q @ pipe.captioner.pool_f.t()
        cand = s_all.topk(10).indices
        mbr = [pipe.captioner.texts.index(c['caption']) for c in pipe.captioner.caption(q, alternatives=0)]
        for i, (seg, cap, kind) in enumerate(jobs):
            rows['q'].append(q[i].half().cpu().numpy()); rows['tok'].append(tok[i].half().cpu().numpy())
            rows['cand'].append(cand[i].cpu().numpy()); rows['mbr'].append(mbr[i])
            rows['cap'].append(cap); rows['vid'].append(vid); rows['split'].append(x['subset'])
            rows['kind'].append(kind); rows['seg'].append(np.asarray(seg, np.float32))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **{k: np.array(v) if k not in ('q', 'tok', 'cand', 'seg') else np.stack(v) for k, v in rows.items()},
             pool=np.array(pipe.captioner.texts))
    kinds = np.array(rows['kind']); split = np.array(rows['split'])
    print('%s: %d segments (train %d, val gt %d, val pred %d)' % (OUT, len(kinds), (split == 'training').sum(),
          ((split == 'validation') & (kinds == 'gt')).sum(), (kinds == 'pred').sum()))


if __name__ == '__main__':
    main()
