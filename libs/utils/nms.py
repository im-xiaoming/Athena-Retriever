# NMS cho đoạn 1D, sửa từ:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
import torch

import nms_1d_cpu


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, min_score, max_num
    ):
        # NMS thường không đổi điểm, nên lọc theo điểm trước được
        is_filtering_by_score = (min_score > 0)
        if is_filtering_by_score:
            valid_mask = scores > min_score
            segs, scores = segs[valid_mask], scores[valid_mask]
            cls_idxs = cls_idxs[valid_mask]

        # gọi nms (C++), trả về chỉ số đã xếp theo điểm giảm dần
        inds = nms_1d_cpu.nms(
            segs.contiguous().cpu(),
            scores.contiguous().cpu(),
            iou_threshold=float(iou_threshold))
        # giữ tối đa max_num đoạn
        if max_num > 0:
            inds = inds[:min(max_num, len(inds))]
        # trả về đoạn / điểm đã sắp xếp
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
        # cấp phát trước chỗ cho kết quả
        dets = segs.new_empty((segs.size(0), 3), device='cpu')
        # gọi soft-NMS (C++), dets chứa đoạn / điểm đã sắp xếp
        inds = nms_1d_cpu.softnms(
            segs.cpu(),
            scores.cpu(),
            dets.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=int(method))
        # giữ tối đa max_num đoạn
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
    """NMS 1D không phân lớp trên segs (N, 2) với scores (N,). Trả về (segs, scores, cls_idxs) của các đoạn
    được giữ, tốt nhất đứng trước, tối đa max_seg_num đoạn. use_soft_nms: soft-NMS kiểu Gaussian (điểm giảm
    dần theo độ chồng lấn, sigma) thay vì bỏ hẳn đoạn chồng lấn quá iou_threshold. cls_idxs đi kèm từng đoạn."""
    # dựa theo cách làm của Detectron2
    num_segs = segs.shape[0]
    # trường hợp không có dự đoán nào
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

    # sắp xếp theo điểm rồi chỉ giữ tối đa max_seg_num đoạn
    _, idxs = new_scores.sort(descending=True)
    max_seg_num = min(max_seg_num, new_segs.shape[0])
    new_segs = new_segs[idxs[:max_seg_num]]
    new_scores = new_scores[idxs[:max_seg_num]]
    new_cls_idxs = new_cls_idxs[idxs[:max_seg_num]]
    return new_segs, new_scores, new_cls_idxs
