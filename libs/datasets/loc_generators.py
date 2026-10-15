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
    """
        Sinh các "điểm" thời gian: những vị trí trên mỗi tầng pyramid mà các head đưa ra dự đoán.

        Mỗi điểm là một hàng (t, cận dưới, cận trên, stride). t là vị trí của điểm trên lưới tầng 0
        (bội số của stride của tầng). Cận dưới / cận trên giới hạn khoảng cách xa nhất từ điểm tới biên
        của đoạn mà tầng này phụ trách, nên đoạn ngắn rơi vào tầng mịn, đoạn dài rơi vào tầng thô.
        Các điểm được tạo một lần cho max_seq_len * max_buffer_len_factor bước, forward() chỉ cắt ra
        số điểm đúng với độ dài thật.
    """
    def __init__(
        self,
        max_seq_len_ori,    # độ dài chuỗi tối đa để tạo sẵn điểm
        max_buffer_len_factor,  # tạo sẵn cho max_seq_len_ori * hệ số này bước
        fpn_levels,         # số tầng pyramid
        scale_factor,       # hệ số giảm độ dài giữa hai tầng
        regression_range,   # khoảng hồi quy của từng tầng (đơn vị bước)
        max_div_factor,     # stride lớn nhất, độ dài phải chia hết cho số này
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
            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float)
            fpn_stride = torch.as_tensor(stride, dtype=torch.float)
            points = torch.arange(0, self.max_seq_len, stride)[:, None]
            # dời điểm nếu cần (mô hình hiện tại không dùng)
            if self.use_offset:
                points += 0.5 * stride
            # ghép thêm khoảng hồi quy và stride vào mỗi điểm
            reg_range = reg_range[None].repeat(points.shape[0], 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)
            # kích thước: T x 4 (t, khoảng hồi quy, stride)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))

        return BufferList(points_list)

    def forward(self, fpn_strides, feats, is_training):
        """Trả về điểm của mọi tầng cho feats (C, T): list các tensor (T / stride, 4). Lúc train và với
        video ngắn thì luôn dùng max_seq_len; video dài hơn lúc suy luận được làm tròn lên bội số của stride lớn nhất."""
        pts_list = []
        max_len = feats.shape[1]
        if is_training:
            max_len = self.max_seq_len_ori
        else:
            if max_len <= self.max_seq_len_ori:
                max_len = self.max_seq_len_ori
            else:
                # làm tròn lên bội số gần nhất của stride lớn nhất
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride

        feat_lens = [int(max_len/stride) for stride in fpn_strides]
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            pts = buffer_pts[:feat_len, :]
            pts_list.append(pts)
        return pts_list 

