# 1D NMS for segments, in numpy (it replaces the C++ extension of ActionFormer, ported from
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/nms.cpp, with the same results)
import numpy as np
import torch


def _overlap(seg, x1, x2, areas):
    """IoU of one segment (x1, x2, area) with every segment of x1 / x2 / areas, float32 as the C++ code."""
    inter = np.maximum(np.float32(0), np.minimum(seg[1], x2) - np.maximum(seg[0], x1))
    return inter / (seg[2] + areas - inter)


def nms(segs, scores, iou_threshold):
    """Plain NMS: indices of the kept segments, best first."""
    x1, x2 = segs[:, 0], segs[:, 1]
    areas = x2 - x1 + np.float32(1e-6)
    order = np.argsort(-scores, kind='stable')
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        ovr = _overlap((x1[i], x2[i], areas[i]), x1[order[1:]], x2[order[1:]], areas[order[1:]])
        order = order[1:][ovr < iou_threshold]
    return np.array(keep, dtype=np.int64)


def soft_nms(segs, scores, iou_threshold, sigma, min_score, method):
    """Soft-NMS: repeatedly take the best remaining segment and lower the scores of the others by their
    overlap with it (method 0: plain NMS, 1: linear, 2: Gaussian with sigma); a segment whose score falls
    below min_score is dropped. Returns the indices of the kept segments in the order they were taken,
    and their (lowered) scores."""
    x1, x2 = segs[:, 0], segs[:, 1]
    areas = x2 - x1 + np.float32(1e-6)
    sc = scores.copy()
    alive = np.arange(len(sc))
    keep, keep_sc = [], []
    while alive.size:
        best = int(np.argmax(sc[alive]))
        i = alive[best]
        keep.append(i)
        keep_sc.append(sc[i])
        alive = np.delete(alive, best)
        ovr = _overlap((x1[i], x2[i], areas[i]), x1[alive], x2[alive], areas[alive])
        if method == 0:
            weight = np.where(ovr >= iou_threshold, np.float32(0), np.float32(1))
        elif method == 1:
            weight = np.where(ovr >= iou_threshold, np.float32(1) - ovr, np.float32(1))
        else:
            weight = np.exp(-(ovr * ovr) / np.float32(sigma))
        sc[alive] *= weight.astype(np.float32)
        alive = alive[sc[alive] >= min_score]
    return np.array(keep, dtype=np.int64), np.array(keep_sc, dtype=np.float32)


def batched_nms(
    segs,
    scores,
    cls_idxs,
    iou_threshold,
    min_score,
    max_seg_num,
    use_soft_nms=True,
    sigma=0.5,
):
    """Class-agnostic 1D NMS of segs (N, 2) with scores (N,). Returns (segs, scores, cls_idxs) of the kept
    segments, best first, at most max_seg_num of them. use_soft_nms: Gaussian soft-NMS (scores decay with the
    overlap, sigma) instead of dropping segments that overlap more than iou_threshold. cls_idxs follow their segment."""
    # no predictions
    if segs.shape[0] == 0:
        return torch.zeros([0, 2]),\
               torch.zeros([0,]),\
               torch.zeros([0,], dtype=cls_idxs.dtype)

    s = segs.detach().cpu().float().numpy()
    p = scores.detach().cpu().float().numpy()
    if use_soft_nms:
        inds, new_p = soft_nms(s, p, iou_threshold, sigma, min_score, 2)
    else:
        # plain NMS does not change the scores, so it can filter by score first
        cand = np.nonzero(p > min_score)[0] if min_score > 0 else np.arange(len(p))
        inds = cand[nms(s[cand], p[cand], iou_threshold)]
        new_p = p[inds]

    # sort by score, keep at most max_seg_num segments
    order = np.argsort(-new_p, kind='stable')[:max_seg_num]
    inds = torch.from_numpy(inds[order])
    return segs.cpu()[inds], torch.from_numpy(new_p[order]), cls_idxs.cpu()[inds]
