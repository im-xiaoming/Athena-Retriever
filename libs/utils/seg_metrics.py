"""Standard class-agnostic localization metrics, next to the repo's own R@t (K = number of GT steps).

R@t in train.py is this repo's convention, not a published one. The metrics here are the ones papers report:

- mAP, ActivityNet style (ActionFormer, TriDet, UniAV, UnAV-100): every prediction of every video (up to
  max_seg_num = 100 per video) ranked together by score; in score order a prediction is a true positive when
  its best still unmatched GT has tIoU >= t (one-to-one), else a false positive; AP is the area under the
  interpolated precision-recall curve. Thresholds 0.5:0.05:0.95 (ActivityNet 1.3) and 0.1:0.1:0.9 (UniAV).
- Dense captioning localization (densevid_eval of ActivityNet Captions / YouCook2; PDVC, Vid2Seq): per video,
  precision = share of the predictions with tIoU > t to some GT, recall = share of the GT with tIoU > t to
  some prediction (strict >, not one-to-one); both averaged over videos and over t in {0.3, 0.5, 0.7, 0.9};
  F1 = harmonic mean of the two averages. The official tool takes every submitted prediction (up to 1000).
  Our model has no event-count head and always returns up to 100, so it is also scored on the top K (oracle
  count, as R@t) and with one score threshold for every video and every dataset (production does not know
  either): chosen on the training videos (best_threshold, train.py --thr-from-train) or, optimistic, the best
  one on the scored set itself.
"""
import numpy as np

TIOU_ANET = np.round(np.arange(0.5, 0.96, 0.05), 2)
TIOU_UNIAV = np.round(np.arange(0.1, 0.91, 0.1), 2)
TIOU_DVC = (0.3, 0.5, 0.7, 0.9)
SCORE_THS = np.round(np.arange(0.05, 0.96, 0.05), 2)   # candidate global score thresholds


def iou_matrix(p, g):
    """(N, 2) x (M, 2) segments -> (N, M) tIoU."""
    p, g = np.asarray(p, float).reshape(-1, 2), np.asarray(g, float).reshape(-1, 2)
    inter = (np.minimum(p[:, None, 1], g[None, :, 1]) - np.maximum(p[:, None, 0], g[None, :, 0])).clip(min=0)
    union = (p[:, 1] - p[:, 0])[:, None] + (g[:, 1] - g[:, 0])[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def interpolated_ap(prec, rec):
    """Area under the precision-recall curve with precision made monotone (ActivityNet's interpolated_prec_rec)."""
    mprec = np.concatenate(([0.0], prec, [0.0]))
    mrec = np.concatenate(([0.0], rec, [1.0]))
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def average_precision(videos, ths):
    """videos: list of (pred (N, 2), scores (N,), gt (M, 2)). Returns {t: AP} pooled over all videos."""
    n_gt = sum(len(g) for _, _, g in videos)
    rows = [(s, v, i) for v, (_, sc, _) in enumerate(videos) for i, s in enumerate(sc)]
    rows.sort(key=lambda r: -r[0])
    ious = [iou_matrix(p, g) if len(p) and len(g) else np.zeros((len(p), len(g))) for p, _, g in videos]
    out = {}
    for t in ths:
        locked = [np.zeros(len(g), bool) for _, _, g in videos]
        tp = np.zeros(len(rows))
        for k, (_, v, i) in enumerate(rows):
            row = ious[v][i]
            for j in np.argsort(-row):          # best GT first, as ActivityNet's evaluator
                if row[j] < t:
                    break
                if not locked[v][j]:
                    locked[v][j] = True
                    tp[k] = 1
                    break
        ctp = np.cumsum(tp)
        prec = ctp / np.arange(1, len(rows) + 1)
        rec = ctp / max(n_gt, 1)
        out[t] = interpolated_ap(prec, rec) if len(rows) else 0.0
    return out


def dvc_pr(videos, ths=TIOU_DVC, top=None, min_score=None):
    """densevid_eval precision / recall, averaged over videos then over ths.

    top: 'k' keeps the top K = number of GT steps per video; min_score keeps scores >= it; neither = all."""
    P, R = [], []
    for p, sc, g in videos:
        p, sc = np.asarray(p, float).reshape(-1, 2), np.asarray(sc, float)
        if top == 'k':
            p = p[:len(g)]
        elif min_score is not None:
            p = p[sc >= min_score]
        m = iou_matrix(p, g) if len(p) and len(g) else np.zeros((len(p), len(g)))
        P.append([float((m > t).any(1).mean()) if len(p) else 0.0 for t in ths])
        R.append([float((m > t).any(0).mean()) if len(g) else 0.0 for t in ths])
    p, r = float(np.mean(P)), float(np.mean(R))
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


def best_threshold(sets):
    """sets: one video list per dataset. The score threshold with the best mean densevid F1 over the sets."""
    return float(max(SCORE_THS, key=lambda s: np.mean([dvc_pr(v, min_score=s)[2] for v in sets])))


def standard_scores(videos, thr=None):
    """All standard metrics of one set, in percent. Keys: AP@0.5, mAP (0.5:0.95), mAP.1-.9 (UniAV),
    dvc P / R / F1 (all predictions), dvc F1@K (top K), dvc F1@best (the best score threshold on this set,
    optimistic; its value in 'dvc best thr') and, when thr is given (chosen elsewhere), dvc P / R / F1@thr."""
    out = {}
    ap = average_precision(videos, sorted(set(TIOU_ANET) | set(TIOU_UNIAV)))
    out['AP@0.5'] = 100 * ap[0.5]
    out['mAP'] = 100 * float(np.mean([ap[t] for t in TIOU_ANET]))
    out['mAP.1-.9'] = 100 * float(np.mean([ap[t] for t in TIOU_UNIAV]))
    p, r, f = dvc_pr(videos)
    out.update({'dvc P': 100 * p, 'dvc R': 100 * r, 'dvc F1': 100 * f})
    out['dvc F1@K'] = 100 * dvc_pr(videos, top='k')[2]
    best = max(((dvc_pr(videos, min_score=s)[2], s) for s in SCORE_THS), key=lambda x: x[0])
    out['dvc F1@best'], out['dvc best thr'] = 100 * best[0], float(best[1])
    if thr is not None:
        p, r, f = dvc_pr(videos, min_score=thr)
        out.update({'dvc P@thr': 100 * p, 'dvc R@thr': 100 * r, 'dvc F1@thr': 100 * f})
    return out
