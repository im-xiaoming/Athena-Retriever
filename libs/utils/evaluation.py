"""Validation scores of the event model: segmentation, captioning, grounding and the standard metrics."""
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .seg_metrics import standard_scores


def iou(a, b):
    """IoU of two segments (start, end)."""
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def mbr_pick(s, pool_n, scale, k=20):
    """Pick a caption by consensus instead of taking the top-scoring one.

    s (N,) scores every caption in the pool. Take the top k, turn their scores into
    probabilities, and return the candidate most similar to the whole group
    (minimum Bayes risk). A caption backed by many strong candidates beats one that
    is on top by chance. Returns (top-1 index, MBR index).
    """
    top = s.topk(k)
    p = F.softmax(scale * top.values, dim=0)
    cand = pool_n[top.indices]
    agree = (cand @ cand.t()) @ p
    return int(top.indices[0]), int(top.indices[agree.argmax()])


def meteor(gts, hyp, timeout=60):
    """METEOR via pycocoevalcap (java), in a separate Python process with a time limit.

    On Colab the java helper can die mid-call while pycocoevalcap holds its lock;
    the object's __del__ then waits on that lock forever. Isolating it in a child
    process means a failure or hang only costs this metric, never the evaluation.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'in.json')
        with open(src, 'w') as f:
            json.dump({'gts': {str(k): v for k, v in gts.items()}, 'hyp': {str(k): v for k, v in hyp.items()}}, f)
        code = ('import json,sys; from pycocoevalcap.meteor.meteor import Meteor; '
                'd=json.load(open(sys.argv[1])); print(Meteor().compute_score(d["gts"], d["hyp"])[0]*100, flush=True)')
        try:
            r = subprocess.run([sys.executable, '-c', code, src], capture_output=True, text=True, timeout=timeout)
            return float(r.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            print('METEOR skipped: timed out after %ds' % timeout, flush=True)
        except (ValueError, IndexError):
            print('METEOR skipped: %s' % (r.stderr.strip().splitlines() or ['no output'])[-1][:200], flush=True)
    return None


@torch.no_grad()
def evaluate(model, loader, device, pool_raw, pool_text, ths=(0.3, 0.5, 0.7), full=False, std=False, thr=None):
    """Score segmentation and captioning on the validation set.

    R@t      : per video keep as many predictions (best first) as it has GT steps; a GT step counts at t
               when one of them has IoU >= t. This repo's own measure (no paper reports it); mIoU the mean best IoU
    Captioning is scored only on GT segments matched by a prediction with IoU >= 0.3.
    cos_gt   : cosine between the predicted segment vector and its GT caption
    cos_rand : the same against a random caption, i.e. the floor
    txt_sim  : how close the chosen caption is to the GT caption, cosine in InternVideo2 text space
    CIDEr    : standard captioning metric, word overlap, independent of any encoder
    G R1@t   : grounding (with the ground head): each GT caption is the query, its best segment
               must overlap the caption's own span with IoU >= t; G mIoU is the mean IoU
    full     : also score variants (top-1, single-point vector, oracle GT spans)
    std      : also the standard localization metrics of papers (mAP, densevid P/R/F1; libs/utils/seg_metrics.py),
               thr the densevid score threshold chosen on the training videos (None: not scored)
    """
    from pycocoevalcap.cider.cider import Cider
    model.eval()
    pool_f = model.embed_head.embed_captions(pool_raw)
    pool_n = F.normalize(pool_raw, dim=-1)   # metrics and the consensus pick: InternVideo2 caption vectors
    scale = float(model.embed_head.logit_scale.exp().clamp(max=100))
    cov = {t: 0 for t in ths}              # GT steps covered by a prediction with IoU >= t
    n_gt = n_pred = 0
    best_ious, cos_gt, cos_rand = [], [], []
    # caption picks, one list per way of picking: txt_sim of every captioned GT step (ret) and the
    # chosen caption texts for CIDEr (hyp), both keyed like gts_txt
    ret = {v: [] for v in ('top1', 'mbr', 'pt_top1', 'oracle')}
    gts_txt, hyp = {}, {v: {} for v in ret}
    gts_or, hyp_or = {}, {}                # oracle: captions picked for the GT spans themselves
    g_ious = []                            # grounding: IoU of every GT caption's predicted span
    videos = [] if std else None           # (predictions, scores, GT) per video for the standard metrics
    g = torch.Generator().manual_seed(0)   # the same random captions for cos_rand in every evaluation
    for batch in loader:
        for x in batch:
            x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
        res = model(batch)
        for r, x in zip(res, batch):
            gts, texts = r['gt_segments'], r['gt_text']
            if videos is not None:
                videos.append(video_record(r))
            if r.get('ground_segments') is not None:
                g_ious += [iou(p, gt) for p, gt in zip(r['ground_segments'].tolist(), gts)]
            # keep as many predictions (best score first) as the video has GT steps
            k = min(len(gts), r['segments'].shape[0])
            preds = r['segments'][:k].tolist()
            emb = r['embeds'][:k].to(device).float()
            emb_pt = r['embeds_pt'][:k].to(device).float()
            n_pred += k
            cap_raw = x['cap_emb'].to(device).float()
            cap_proj = model.embed_head.embed_captions(cap_raw)
            cap_n = F.normalize(cap_raw, dim=-1)
            if full:
                # caption the GT spans themselves: measures captioning alone
                for gi in range(len(gts)):
                    i = mbr_pick(pool_f @ r['embeds_gt'][gi].to(device).float(), pool_n, scale)[1]
                    ret['oracle'].append(float(pool_n[i] @ cap_n[gi]))
                    okey = len(gts_or)
                    gts_or[okey] = [texts[gi]]
                    hyp_or[okey] = [pool_text[i]]
            for gi, gt in enumerate(gts):
                n_gt += 1
                if not preds:
                    best_ious.append(0.0)
                    continue
                vals = [iou(p, gt) for p in preds]
                j = int(np.argmax(vals))   # the prediction that matches this GT step best
                best = vals[j]
                best_ious.append(best)
                for t in ths:
                    cov[t] += (best >= t)
                if best < 0.3:             # captioning is scored on matched steps only
                    continue
                q = F.normalize(emb[j], dim=-1)
                cos_gt.append(float(q @ cap_proj[gi]))
                ridx = int(torch.randint(len(pool_f), (1,), generator=g))
                cos_rand.append(float(q @ pool_f[ridx]))
                key = len(gts_txt)
                gts_txt[key] = [texts[gi]]
                t1, mb = mbr_pick(pool_f @ q, pool_n, scale)
                picks = {'top1': t1, 'mbr': mb}
                if full:
                    picks['pt_top1'] = int((pool_f @ F.normalize(emb_pt[j], dim=-1)).argmax())
                for v, i in picks.items():
                    ret[v].append(float(pool_n[i] @ cap_n[gi]))
                    hyp[v][key] = [pool_text[i]]
    model.train()
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    cider = lambda v: Cider().compute_score(gts_txt, hyp[v])[0] * 100 if hyp[v] else 0.0
    out = {'R@%.1f' % t: 100.0 * cov[t] / max(n_gt, 1) for t in ths}
    out['mIoU'] = 100.0 * mean(best_ious)
    out['seg/vid'] = n_pred / max(len(loader.dataset), 1)
    out['cos_gt'] = mean(cos_gt)
    out['cos_rand'] = mean(cos_rand)
    out['txt_sim'] = mean(ret['mbr'])
    out['CIDEr'] = cider('mbr')
    if videos is not None:
        out.update(standard_scores(videos, thr))
    if g_ious:
        for t in (0.5, 0.7):
            out['G R1@%.1f' % t] = 100.0 * mean([i >= t for i in g_ious])
        out['G mIoU'] = 100.0 * mean(g_ious)
    if full:
        for v in ('top1', 'pt_top1'):
            out['txt_sim[%s]' % v] = mean(ret[v])
            out['CIDEr[%s]' % v] = cider(v)
        m = meteor(gts_txt, hyp['mbr'])
        if m is not None:
            out['METEOR'] = m
        out['txt_sim[oracle]'] = mean(ret['oracle'])
        out['CIDEr[oracle]'] = Cider().compute_score(gts_or, hyp_or)[0] * 100
        out['n_captioned'] = len(gts_txt)
    return out


def video_record(r):
    """(predictions (N, 2) s, scores (N,), GT (M, 2) s) of one video, for libs/utils/seg_metrics.py."""
    cpu = lambda x: torch.as_tensor(x).detach().float().cpu().numpy()
    return cpu(r['segments']).reshape(-1, 2), cpu(r['scores']).reshape(-1), cpu(r['gt_segments']).reshape(-1, 2)


@torch.no_grad()
def collect_predictions(model, loader, device, ids=None):
    """(predictions, scores, GT) of every video of the loader (video_record);
    ids (list): filled with the video ids and GT captions, in the same order."""
    model.eval()
    w_ground, model.w_ground = model.w_ground, 0   # no grounding needed: skip its passes
    videos = []
    for batch in loader:
        for x in batch:
            x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
        for r in model(batch):
            videos.append(video_record(r))
            if ids is not None:
                ids.append((r['video_id'], list(r['gt_text'][:len(r['gt_segments'])])))
    model.w_ground = w_ground
    model.train()
    return videos


def dump_predictions(model, loader, device, out_dir):
    """Write the validation set in the densevid_eval formats, for the official tool
    (experiments/scripts/2026-10-13_std-metrics/official_dvc.py): htstep_gt.json {video: {timestamps, sentences}}
    and htstep_pred.json {video: [{timestamp, score}, ...] best score first}."""
    os.makedirs(out_dir, exist_ok=True)
    ids = []
    videos = collect_predictions(model, loader, device, ids)
    gt = {v: {'timestamps': g.tolist(), 'sentences': t} for (v, t), (_, _, g) in zip(ids, videos)}
    pred = {v: [{'timestamp': p, 'score': s} for p, s in zip(pr.tolist(), sc.tolist())]
            for (v, _), (pr, sc, _) in zip(ids, videos)}
    for kind, obj in (('gt', gt), ('pred', pred)):
        with open(os.path.join(out_dir, 'htstep_%s.json' % kind), 'w') as f:
            json.dump(obj, f)
    print('Dumped       : %d videos -> %s' % (len(ids), out_dir), flush=True)
