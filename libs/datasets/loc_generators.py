import torch
from torch import nn
from torch.nn import functional as F


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers

    Taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/anchor_generator.py
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

class PointGenerator(nn.Module):
    """
        A generator for temporal "points": the positions on every pyramid level where the heads predict.

        Each point is a row (t, regression range lo, regression range hi, stride): t is its position on
        the level-0 grid (multiples of the level's stride); the range bounds the larger of the point's
        two distances (in grid steps) to the boundaries of a segment this level is responsible for, so
        short segments go to fine levels and long ones to coarse levels. Points are built once for
        max_seq_len * max_buffer_len_factor steps; forward() returns the first ones for the actual length.
    """
    def __init__(
        self,
        max_seq_len_ori,    # max sequence length that the generator will buffer
        max_buffer_len_factor, 
        fpn_levels,         # number of fpn levels
        scale_factor,       # scale factor between two fpn levels
        regression_range,   # regression range (on feature grids)
        max_div_factor,     
        use_offset=False    # if to align the points at grid centers
    ):
        super().__init__()
        # sanity check, # fpn levels and length divisible
        assert len(regression_range) == fpn_levels
        max_seq_len = max_seq_len_ori * max_buffer_len_factor 
        assert max_seq_len % scale_factor**(fpn_levels - 1) == 0

        # save params
        self.max_seq_len_ori = max_seq_len_ori  
        self.max_seq_len = max_seq_len 
        self.fpn_levels = fpn_levels
        self.scale_factor = scale_factor
        self.regression_range = regression_range
        self.use_offset = use_offset
        self.max_div_factor = max_div_factor  

        # generate all points and buffer the list
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        points_list = []
        # loop over all points at each pyramid level
        for l in range(self.fpn_levels):
            stride = self.scale_factor ** l
            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float)
            fpn_stride = torch.as_tensor(stride, dtype=torch.float)
            points = torch.arange(0, self.max_seq_len, stride)[:, None]
            # add offset if necessary (not in our current model)
            if self.use_offset:
                points += 0.5 * stride
            # pad the time stamp with additional regression range / stride
            reg_range = reg_range[None].repeat(points.shape[0], 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)
            # size: T x 4 (ts, reg_range, stride)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))

        return BufferList(points_list)

    def forward(self, fpn_strides, feats, is_training):
        """Points of every level for feats (C, T): a list of (T / stride, 4) tensors. Training and
        short videos always use max_seq_len; a longer input at inference is padded to a multiple of
        the largest stride, as the model pads it."""
        pts_list = []
        max_len = feats.shape[1]
        if is_training:
            max_len = self.max_seq_len_ori
        else:
            if max_len <= self.max_seq_len_ori:
                max_len = self.max_seq_len_ori
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride

        feat_lens = [int(max_len/stride) for stride in fpn_strides]
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            assert feat_len <= buffer_pts.shape[0], "Reached max buffer length for point generator"
            pts = buffer_pts[:feat_len, :]
            pts_list.append(pts)
        return pts_list 

