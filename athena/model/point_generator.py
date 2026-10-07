"""Temporal anchor points per FPN level (copied from libs/datasets/loc_generators.py)."""
import torch
from torch import nn


class BufferList(nn.Module):
    """Like nn.ParameterList, but for non-persistent buffers."""

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class PointGenerator(nn.Module):
    """Points (t, reg_range_lo, reg_range_hi, stride) for every FPN level."""

    def __init__(self, max_seq_len_ori, max_buffer_len_factor, fpn_levels, scale_factor,
                 regression_range, max_div_factor, use_offset=False):
        super().__init__()
        assert len(regression_range) == fpn_levels
        max_seq_len = max_seq_len_ori * max_buffer_len_factor
        assert max_seq_len % scale_factor ** (fpn_levels - 1) == 0
        self.max_seq_len_ori = max_seq_len_ori
        self.max_seq_len = max_seq_len
        self.fpn_levels = fpn_levels
        self.scale_factor = scale_factor
        self.regression_range = regression_range
        self.use_offset = use_offset
        self.max_div_factor = max_div_factor
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        points_list = []
        for l in range(self.fpn_levels):
            stride = self.scale_factor ** l
            reg_range = torch.as_tensor(self.regression_range[l], dtype=torch.float)
            fpn_stride = torch.as_tensor(stride, dtype=torch.float)
            points = torch.arange(0, self.max_seq_len, stride)[:, None]
            if self.use_offset:
                points += 0.5 * stride
            reg_range = reg_range[None].repeat(points.shape[0], 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))
        return BufferList(points_list)

    def forward(self, fpn_strides, feat_len):
        """Inference-time points for a sequence of feat_len steps (same rule as training code)."""
        max_len = feat_len
        if max_len <= self.max_seq_len_ori:
            max_len = self.max_seq_len_ori
        else:   # pad to the next size divisible by the largest stride
            max_len = (max_len + self.max_div_factor - 1) // self.max_div_factor * self.max_div_factor
        return [pts[:int(max_len / stride), :] for stride, pts in zip(fpn_strides, self.buffer_points)]
