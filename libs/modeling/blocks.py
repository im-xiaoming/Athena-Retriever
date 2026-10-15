import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


def Linear(in_features, out_features, bias=True):
    """nn.Linear với trọng số khởi tạo Xavier-uniform và bias bằng 0."""
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class MaskedConv1D(nn.Module):
    """
    Conv1D có mask: đầu ra ở các bước padding bị đặt về 0. Cách gọi giống Conv1d,
    nhưng chỉ hỗ trợ một phần các kiểu conv 1D.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # kernel lẻ và padding = kernel // 2 để đầu ra thẳng hàng với đầu vào
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # bước nhảy
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # nếu có bias thì đặt về 0
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: (batch, số kênh, độ dài chuỗi)
        # mask: (batch, 1, độ dài chuỗi), kiểu bool
        B, C, T = x.size()
        # độ dài đầu vào phải chia hết cho stride
        assert T % self.stride == 0
        # tính conv
        out_conv = self.conv(x)
        # tính mask cho đầu ra
        if self.stride > 1:
            # giảm mask theo stride bằng nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=T//self.stride,
                mode='nearest'
            )
        else:
            # giữ nguyên mask
            out_mask = mask.to(x.dtype)

        if out_mask.shape[-1] != out_conv.shape[-1]:
            out_mask = out_mask[:, None, None]
        # nhân đầu ra với mask, mask không cần gradient
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm cho tensor dạng (B, C, T): chuẩn hóa theo trục kênh C
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        # chuẩn hóa theo trục kênh C
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)
        # nhân weight, cộng bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out

# hàm phụ cho các block Transformer
def get_sinusoid_encoding(n_position, d_hid):
    ''' Bảng position encoding dạng sin / cos '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # trả về tensor kích thước (1, C, T)
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)

class MaskedMHCA(nn.Module):
    """
    Multi-head attention có conv và mask (từ ActionFormer)

    So với MHA thường thì có thêm một conv depthwise trên query, key, value. Conv này dùng để
    (1) mã hóa vị trí tương đối (thay cho position encoding);
    (2) giảm độ dài feature khi cần (stride 2 -> các tầng pyramid);
    (3) khớp số kênh.

    Với cách cài hiện tại, feature sau khi giảm độ dài sẽ thẳng hàng với mỗi bước thứ s+1
    (s là stride), nhờ vậy dễ nội suy position embedding tương ứng.

    Sửa từ https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # số kênh đầu ra
        n_head,          # số head attention
        n_qx_stride=1,   # stride giảm độ dài cho query và đầu vào
        n_kv_stride=1,   # stride giảm độ dài cho key và value
        attn_pdrop=0.0,  # dropout trên ma trận attention
        proj_pdrop=0.0,  # dropout ở phép chiếu đầu ra
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # các phép conv / pooling
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # conv depthwise cho query. Stride lấy theo n_kv_stride giống bản gốc, ở đây hai giá trị
        # luôn bằng nhau nên không sao
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # conv depthwise 1D
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # conv depthwise cho key, value
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # conv depthwise 1D
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # phép chiếu key, query, value cho mọi head
        # không cần mask ở đây vì mask sẽ áp lên attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization (dropout)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # phép chiếu đầu ra
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x1, x2, mask):
        # x1: luồng cho key và value; x2: luồng cho query (self-attention thì x1 chính là x2).
        # Đầu ra có một hàng cho mỗi bước của query.
        # x: (batch, số kênh, độ dài chuỗi)
        # mask: (batch, 1, độ dài chuỗi), kiểu bool
        B, C, T = x1.size()

        # conv cho query -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x2, mask)
        q = self.query_norm(q)
        # conv cho key, value -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x1, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x1, mask)
        v = self.value_norm(v)

        # chiếu sang q, k, v
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # đưa trục head lên cạnh trục batch
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)

        if T == mask.shape[-1]:
            # không cho q nhìn vào các bước padding
            att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
            # softmax trên attention
            att = F.softmax(att, dim=-1) 
        else:
            att = F.softmax(att, dim=-1)  
            att = att * kv_mask[:, :, :, None].to(att.dtype) 

        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # ghép đầu ra của các head lại
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # chiếu đầu ra, nhân mask
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)

        return out, qx_mask

class TransformerBlock(nn.Module):
    """
    Block Transformer đơn giản (LayerNorm đặt trước attention và MLP)
    Sửa từ https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,                # số kênh đầu vào
        n_head,                # số head attention
        n_ds_strides=(1, 1),   # stride giảm độ dài cho (q và x, k và v)
        n_out=None,            # số kênh đầu ra, None thì bằng số kênh đầu vào
        n_hidden=None,         # số kênh lớp ẩn của MLP
        act_layer=nn.GELU,     # hàm kích hoạt trong MLP, mặc định GELU
        attn_pdrop=0.0,        # dropout trên ma trận attention
        proj_pdrop=0.0,        # dropout ở phép chiếu / MLP
        path_pdrop=0.0,        # tỉ lệ drop path
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # LayerNorm cho tensor dạng (B C T)
        self.ln11 = LayerNorm(n_embd)
        self.ln12 = LayerNorm(n_embd)
        self.n_embd = n_embd

        self.attn = MaskedMHCA(
            n_embd,
            n_head,
            n_qx_stride=n_ds_strides[0],
            n_kv_stride=n_ds_strides[1],
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop
            )

        # nhánh tắt (skip): giảm độ dài bằng max pooling nếu stride > 1
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # MLP hai lớp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # mặc định
        if n_out is None:
            n_out = n_embd

        # mạng feed-forward. UniAV gốc có ba cái cho ba task (TAL, AVEL, SED), ở đây chỉ dùng của TAL
        self.ln2 = LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x1, x2, mask, pos_embd=None):
        # x1: luồng được cập nhật (cho key, value và nhánh residual); x2: luồng cho query
        # (self-attention thì là chính x1): out = x1 + Attn(q=x2, k=x1, v=x1), rồi qua MLP.
        # transformer kiểu pre-LN: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln11(x1), self.ln12(x2), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x1) * out_mask_float + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)

        # nếu có pos_embd thì cộng vào đầu ra
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        
        return out, out_mask 

class Scale(nn.Module):
    """
    Nhân đầu ra hồi quy với một hằng số học được
    """
    def __init__(self, init_value=1.0):
        """
        init_value : giá trị ban đầu của hằng số
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        đầu vào -> scale * đầu vào
        """
        return x * self.scale

# đoạn dưới sửa từ
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic depth: bỏ cả nhánh residual của từng mẫu với xác suất drop_prob.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # chạy được với tensor nhiều chiều, không chỉ ConvNet 2D
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # đưa về 0 / 1
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop path (stochastic depth) cho từng mẫu, dùng trên nhánh chính của block residual."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop path cho từng mẫu, có thêm hệ số scale theo từng kênh (khởi tạo rất nhỏ, 1e-4)
    Xem: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):

        return drop_path(self.scale * x, self.drop_prob, self.training)

