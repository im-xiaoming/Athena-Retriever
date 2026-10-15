import torch
from torch.nn import functional as F

@torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Focal loss của RetinaNet (https://arxiv.org/abs/1708.02002), lấy từ
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    inputs: logit, shape bất kỳ
    targets: nhãn 0 / 1, cùng shape với inputs
    alpha: hệ số cân bằng giữa mẫu dương và mẫu âm (mặc định 0.25)
    gamma: số mũ của (1 - p_t), càng lớn thì mẫu dễ càng bị giảm trọng số
    reduction: 'none' | 'mean' | 'sum'
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    class_aware: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU loss (Zheng và cộng sự, https://arxiv.org/abs/1911.08287)

    Mỗi event 1D được biểu diễn bằng một điểm c và hai khoảng cách tới hai biên:
    (t1, t2) = (c - o_1, c + o_2) với o_i >= 0. Dự đoán và nhãn dùng chung điểm c.

    Code tham khảo từ
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    input_offsets / target_offsets: (N, 2) khoảng cách tới hai biên
    reduction: 'none' | 'mean' | 'sum'
    eps: số nhỏ để tránh chia cho 0
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # khoảng cách phải không âm
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    if class_aware:
        mask = torch.logical_or((target_offsets[:, :, 0] > 0), (target_offsets[:, :, 1] > 0)) 
        input_offsets = input_offsets[mask]
        target_offsets = target_offsets[mask]
        
    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # phần giao
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # tính iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # đoạn nhỏ nhất bao cả hai
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # độ lệch giữa hai tâm
    rho = 0.5 * (rp - lp - rg + lg)

    # tính diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
