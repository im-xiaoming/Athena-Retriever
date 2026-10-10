"""Class-agnostic Gaussian soft-NMS in numpy, a line-by-line port of libs/utils/csrc/nms_cpu.cpp
(softnms_1d_cpu, method 2) plus the sorting/truncation done by libs/utils/nms.py:batched_nms.

The API therefore needs no compiled nms_1d_cpu extension. Note that the Gaussian method
ignores the IoU threshold: every overlap decays the score by exp(-iou^2 / sigma).
"""
import numpy as np


def soft_nms(segs, scores, sigma=0.5, min_score=0.001, max_num=100):
    """segs (N, 2), scores (N,) -> (segs, scores, original indices), best first, at most max_num."""
    segs = np.asarray(segs, dtype=np.float32)
    n = segs.shape[0]
    if n == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    x1, x2 = segs[:, 0].copy(), segs[:, 1].copy()
    sc = np.asarray(scores, dtype=np.float32).copy()
    areas = (x2 - x1 + np.float32(1e-6)).astype(np.float32)
    inds = np.arange(n, dtype=np.int64)
    dets = np.empty((n, 3), np.float32)
    nsegs = n   # entries [0, nsegs) are alive; [0, i) are already picked, in order
    for i in range(n):
        if i >= nsegs:
            break
        # pick the best remaining segment and swap it into position i
        max_pos = i + int(np.argmax(sc[i:nsegs])) if nsegs > i else i
        # the C++ loop keeps the first maximum: argmax does too
        ix1, ix2, iscore, iarea, iind = x1[max_pos], x2[max_pos], sc[max_pos], areas[max_pos], inds[max_pos]
        dets[i] = (ix1, ix2, iscore)
        x1[max_pos], x2[max_pos], sc[max_pos], areas[max_pos], inds[max_pos] = x1[i], x2[i], sc[i], areas[i], inds[i]
        x1[i], x2[i], sc[i], areas[i], inds[i] = ix1, ix2, iscore, iarea, iind
        # decay the score of every remaining segment by its overlap with the pick
        pos = i + 1
        while pos < nsegs:
            inter = max(np.float32(0.0), min(ix2, x2[pos]) - max(ix1, x1[pos]))
            ovr = inter / (iarea + areas[pos] - inter)
            sc[pos] = sc[pos] * np.float32(np.exp(-(ovr * ovr) / sigma))
            if sc[pos] < min_score:   # drop: swap in the last live segment
                last = nsegs - 1
                x1[pos], x2[pos], sc[pos], areas[pos], inds[pos] = x1[last], x2[last], sc[last], areas[last], inds[last]
                nsegs -= 1
                pos -= 1
            pos += 1
    keep = min(nsegs, max_num) if max_num > 0 else nsegs
    out_segs, out_scores, out_idx = dets[:keep, :2], dets[:keep, 2], inds[:keep]
    order = np.argsort(-out_scores, kind='stable')
    return out_segs[order], out_scores[order], out_idx[order]
