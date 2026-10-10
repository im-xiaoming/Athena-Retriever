# Functions for 1D NMS, modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
import torch

import nms_1d_cpu


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, min_score, max_num
    ):
        # vanilla nms will not change the score, so we can filter segs first
        is_filtering_by_score = (min_score > 0)
        if is_filtering_by_score:
            valid_mask = scores > min_score
            segs, scores = segs[valid_mask], scores[valid_mask]
            cls_idxs = cls_idxs[valid_mask]

        # nms op; return inds that is sorted by descending order
        inds = nms_1d_cpu.nms(
            segs.contiguous().cpu(),
            scores.contiguous().cpu(),
            iou_threshold=float(iou_threshold))
        # cap by max number
        if max_num > 0:
            inds = inds[:min(max_num, len(inds))]
        # return the sorted segs / scores
        sorted_segs = segs[inds]
        sorted_scores = scores[inds]
        sorted_cls_idxs = cls_idxs[inds]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


class SoftNMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, sigma, min_score, method, max_num
    ):
        # pre allocate memory for sorted results
        dets = segs.new_empty((segs.size(0), 3), device='cpu')
        # softnms op, return dets that stores the sorted segs / scores
        inds = nms_1d_cpu.softnms(
            segs.cpu(),
            scores.cpu(),
            dets.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=int(method))
        # cap by max number
        if max_num > 0:
            n_segs = min(len(inds), max_num)
        else:
            n_segs = len(inds)
        sorted_segs = dets[:n_segs, :2]
        sorted_scores = dets[:n_segs, 2]
        sorted_cls_idxs = cls_idxs[inds]
        sorted_cls_idxs = sorted_cls_idxs[:n_segs]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


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
    """Class-agnostic 1-D NMS over segs (N, 2) with scores (N,) -> (segs, scores, cls_idxs) of the kept
    segments, best first, at most max_seg_num. use_soft_nms: Gaussian soft-NMS (scores decay with
    overlap, sigma) instead of dropping overlaps above iou_threshold. cls_idxs rides along with the
    segments (the event model passes the segment indices to know which ones were kept)."""
    # Based on Detectron2 implementation,
    num_segs = segs.shape[0]
    # corner case, no prediction outputs
    if num_segs == 0:
        return torch.zeros([0, 2]),\
               torch.zeros([0,]),\
               torch.zeros([0,], dtype=cls_idxs.dtype)

    if use_soft_nms:
        new_segs, new_scores, new_cls_idxs = SoftNMSop.apply(
            segs, scores, cls_idxs, iou_threshold,
            sigma, min_score, 2, max_seg_num
        )
    else:
        new_segs, new_scores, new_cls_idxs = NMSop.apply(
            segs, scores, cls_idxs, iou_threshold,
            min_score, max_seg_num
        )

    # sort based on scores and return
    # truncate the results based on max_seg_num
    _, idxs = new_scores.sort(descending=True)
    max_seg_num = min(max_seg_num, new_segs.shape[0])
    new_segs = new_segs[idxs[:max_seg_num]]
    new_scores = new_scores[idxs[:max_seg_num]]
    new_cls_idxs = new_cls_idxs[idxs[:max_seg_num]]
    return new_segs, new_scores, new_cls_idxs
