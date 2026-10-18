from __future__ import annotations
import torch
from torch import nn

from .datasets import register_generator



class BufferList(nn.Module):
    """
    Giống nn.ParameterList nhưng dành cho buffer

    Lấy từ https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/anchor_generator.py
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # buffer không lưu vào checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

@register_generator('point')
class PointGenerator(nn.Module):
    def __init__(
        self,
        regression_range: list[list[int]],
        max_seq_len_ori: int = 256,    # độ dài chuỗi tối đa để tạo sẵn điểm
        max_buffer_len_factor: float = 1.0,  # tạo sẵn cho max_seq_len_ori * hệ số này bước
        fpn_levels: int = 6,         # số tầng pyramid
        scale_factor: int = 2,       # hệ số giảm độ dài giữa hai tầng
        max_div_factor: int = 32,     # stride lớn nhất, độ dài phải chia hết cho số này
        use_offset=False    # có dời điểm vào giữa ô lưới không
    ):
        super().__init__()
        # kiểm tra số tầng và độ dài chia hết
        assert len(regression_range) == fpn_levels
        max_seq_len = max_seq_len_ori * max_buffer_len_factor 
        assert max_seq_len % scale_factor**(fpn_levels - 1) == 0

        # lưu tham số
        self.max_seq_len_ori = max_seq_len_ori  
        self.max_seq_len = max_seq_len 
        self.fpn_levels = fpn_levels
        self.scale_factor = scale_factor
        self.regression_range = regression_range
        self.use_offset = use_offset
        self.max_div_factor = max_div_factor  

        # tạo sẵn toàn bộ điểm
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        points_list = []
        # duyệt từng tầng pyramid
        for l in range(self.fpn_levels):
            stride = self.scale_factor ** l
            reg_range = torch.as_tensor(self.regression_range[l], dtype=torch.float32) # (2,)
            fpn_stride = torch.as_tensor(stride, dtype=torch.float32)
            points = torch.arange(0, self.max_seq_len, stride)[:, None] # (N, 1)
            # dời điểm nếu cần (mô hình hiện tại không dùng)
            if self.use_offset:
                points += 0.5 * stride
            # ghép thêm khoảng hồi quy và stride vào mỗi điểm
            #               (1, 2)          (N, 2)
            reg_range = reg_range[None].repeat(points.shape[0], 1) # (N, 2)
            #               (1, 1)          (N, 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1) # (N, 1)
            # kích thước: T x 4 (t, khoảng hồi quy, stride)
            #                                       (N, 4)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1)) # (N, 4)

        return BufferList(points_list)

    def forward(self, fpn_strides, feats, is_training) -> list[torch.Tensor]:
        """Trả về điểm của mọi tầng cho feats (C, T): list các tensor (T / stride, 4). Lúc train và với
        video ngắn thì luôn dùng max_seq_len; video dài hơn lúc suy luận được làm tròn lên bội số của stride lớn nhất."""
        pts_list = []
        max_len = feats.shape[1] # T
        if is_training:
            max_len = self.max_seq_len_ori
        else:
            if max_len <= self.max_seq_len_ori:
                max_len = self.max_seq_len_ori
            else:
                # làm tròn lên bội số gần nhất của stride lớn nhất
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride

        feat_lens = [int(max_len/stride) for stride in fpn_strides] # [256, 128, 64, 32, 16, 8]
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            pts = buffer_pts[:feat_len, :]
            pts_list.append(pts)
        return pts_list

