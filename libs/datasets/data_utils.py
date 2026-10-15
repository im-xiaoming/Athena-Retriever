import copy
import random
import numpy as np
import torch
from torch.nn import functional as F


def trivial_batch_collator(batch):
    """
        Không gộp gì cả: một batch chỉ là list các dict
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Đặt lại seed ngẫu nhiên cho từng worker của DataLoader
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)

def truncate_feats(
    data_dict,           # dict của một video (xem bên dưới)
    max_seq_len,         # độ dài tối đa (số bước)
    trunc_thresh,        # một đoạn GT được giữ nếu ít nhất tỉ lệ này của nó nằm trong cửa sổ
    offset,              # 0.5: bước i phủ khoảng [i - 0.5, i + 0.5]
    crop_ratio,          # (0.9, 1.0): cửa sổ dài từ 90% đến 100% video
    max_num_trials=200,  # số lần thử tối đa để tìm cửa sổ còn ít nhất một đoạn
):
    """
    Tăng cường dữ liệu lúc train: cắt ngẫu nhiên một cửa sổ của feature và cắt các đoạn GT theo cửa sổ đó.

    Cửa sổ dài từ crop_ratio[0] đến crop_ratio[1] độ dài video. Một đoạn được giữ nếu ít nhất trunc_thresh
    của nó nằm trong cửa sổ. Thử tối đa max_num_trials lần cho tới khi còn ít nhất một đoạn.

    data_dict = {'video_id'        : str
                 'feats'           : {'visual': Tensor C x T, 'audio': Tensor C x T}
                 'segments'        : Tensor N x 2 (đơn vị bước)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : float
                 'feat_num_frames' : float}
    """
    # lấy thông tin; feats = {'visual': C x T, 'audio': C x T}
    feat_len = data_dict['feats']['visual'].shape[1]

    num_segs = data_dict['segments'].shape[0]

    # chọn ngẫu nhiên độ dài cửa sổ trong [l, r]
    max_seq_len = random.randint(
        max(round(crop_ratio[0] * feat_len), 1),
        min(round(crop_ratio[1] * feat_len), feat_len),
    )
    # cửa sổ dài bằng cả video thì không cần cắt
    if feat_len == max_seq_len:
        return data_dict

    # còn lại thì copy dict ra để sửa
    data_dict = copy.deepcopy(data_dict)

    # thử vài lần cho tới khi cửa sổ còn ít nhất một đoạn
    for _ in range(max_num_trials):

        # chọn ngẫu nhiên vị trí cửa sổ
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # tính phần giao giữa cửa sổ và từng đoạn
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0] - offset, data_dict['segments'][:, 0]) 
        right = torch.minimum(window[:, 1] + offset, data_dict['segments'][:, 1])  
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # chỉ giữ các đoạn có phần giao vượt ngưỡng
        seg_idx = (inter_ratio >= trunc_thresh)

        # có ít nhất một đoạn thì dừng
        if seg_idx.sum().item() > 0:
            break

    data_dict['feats']['visual'] = data_dict['feats']['visual'][:, st:ed].clone()
    data_dict['feats']['audio'] = data_dict['feats']['audio'][:, st:ed].clone()
    # segments: N x 2, đơn vị bước
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # dời mốc thời gian theo điểm bắt đầu cửa sổ
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict


def label_points(
    points,       # list các tầng, mỗi tầng (T_l, 4): (t, cận dưới, cận trên của khoảng hồi quy, stride)
    gt_segments,  # (N, 2) các đoạn GT, đơn vị bước
    gt_labels,    # (N,) chỉ số caption của từng đoạn trong video
    num_classes,  # số caption tối đa trong một video (16)
):
        """Nhãn train cho mọi điểm của pyramid (theo ActionFormer).

        Một điểm là dương cho một đoạn GT nếu nó nằm trong đoạn và khoảng cách xa nhất từ nó tới hai biên
        nằm trong khoảng hồi quy của tầng. Nếu nhiều đoạn cùng thỏa thì lấy đoạn ngắn nhất.
        Trả về, với P là tổng số điểm của mọi tầng:
          cls_targets (P, num_classes)  one-hot chỉ số caption của đoạn chứa điểm, toàn 0 nếu là nền
          reg_targets (P, 2)            khoảng cách tới đầu và cuối đoạn, chia cho stride
        """
        # nối điểm của mọi tầng: List[T x 4] -> F T x 4
        concat_points = torch.cat(points, dim=0)
        num_pts = concat_points.shape[0]
        num_gts = gt_segments.shape[0]

        # độ dài của mọi đoạn -> F T x N
        lens = gt_segments[:, 1] - gt_segments[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # khoảng cách từ mỗi điểm tới hai biên của mỗi đoạn -> F T x N x 2
        gt_segs = gt_segments[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        # điểm nằm trong đoạn GT
        inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # giới hạn khoảng hồi quy theo tầng của điểm
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))

        # nếu một điểm vẫn thuộc nhiều đoạn thì chọn đoạn ngắn nhất (dễ hồi quy nhất)
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)
        # trường hợp nhiều đoạn có độ dài gần như bằng nhau
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(gt_labels, num_classes).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # tránh trường hợp nhiều đoạn GT trùng nhãn và trùng biên
        cls_targets.clamp_(min=0.0, max=1.0)

        # lấy khoảng cách tới đoạn được chọn
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # chia cho stride của tầng
        reg_targets /= concat_points[:, 3, None]
        return cls_targets, reg_targets
