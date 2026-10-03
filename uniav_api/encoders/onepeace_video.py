"""ONE-PEACE video backbone (Kinetics-400 fine-tuned), free of mmaction/mmcv.
Copied from ONEPEACE_extract_embd_code/One_Peace_FE/onepeace_video_backbone.py.

Source: https://github.com/OFA-Sys/ONE-PEACE/blob/main/one_peace_vision/video/mmaction_custom/models/backbones/onepeace.py
(Apache 2.0, OFA-Sys). Changes from the original:
  - no mmcv / mmaction / timm dependency (only needed for training and weight init);
  - no position-embedding interpolation (the K400 checkpoint is already 256x256, bucket 16);
  - batch-first tensors (B, L, C) instead of (L, B, C), fewer transposes between layers;
  - attention through F.scaled_dot_product_attention on 4D tensors with a broadcast relative
    position bias; the maths is unchanged;
  - a loader for onepeace_video_k400.pth and a clip-feature helper.
Module and parameter names are unchanged so the state_dict matches the checkpoint exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

# image normalisation used for K400 fine-tuning (configs/recognition/onepeace_k400.py), RGB order
IMG_MEAN = (122.771, 116.746, 104.094)
IMG_STD = (68.5, 66.632, 70.323)


class Adapter(nn.Module):
    """Bottleneck MLP (AIM adapter): D -> D * mlp_ratio -> D, optional skip."""

    def __init__(self, d_features: int, mlp_ratio: float = 0.25, skip_connect: bool = True):
        super().__init__()
        self.skip_connect = skip_connect
        d_hidden = int(d_features * mlp_ratio)
        self.act = nn.GELU()
        self.D_fc1 = nn.Linear(d_features, d_hidden)
        self.D_fc2 = nn.Linear(d_hidden, d_features)

    def forward(self, x: Tensor) -> Tensor:
        xs = self.D_fc2(self.act(self.D_fc1(x)))
        return x + xs if self.skip_connect else xs


def make_image_bucket_position(bucket_size: int, num_relative_distance: int) -> Tensor:
    """2D relative position bias index for (bucket_size^2 + 1) tokens (with CLS)."""
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, h, w
    coords_flatten = torch.flatten(coords, 1)  # 2, h*w
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(
        size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype
    )
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # h*w, h*w
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index  # h*w+1, h*w+1


class LayerNorm2D(nn.Module):
    """Channel-wise LayerNorm for image tensors (B, C, H, W)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GeGLU(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.wi_0 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.wi_1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.wi_0(x)) * self.wi_1(x)


class ImageAdaptor(nn.Module):
    """Patch embedding (hierarchical conv stride 16) + CLS + position/temporal embedding."""

    def __init__(
        self,
        attention_heads: int = 24,
        bucket_size: int = 16,
        num_frames: int = 16,
        dropout: float = 0.0,
        embed_dim: int = 1536,
        shared_rp_bias: bool = True,
    ):
        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.embed_images = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, kernel_size=4, stride=4),
            LayerNorm2D(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2D(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
        )

        scale = embed_dim**-0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(1, 1, embed_dim))

        self.bucket_size = bucket_size
        self.num_frames = num_frames
        self.pos_embed = nn.Parameter(scale * torch.randn(bucket_size**2 + 1, embed_dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        self.shared_rp_bias = shared_rp_bias
        if shared_rp_bias:
            num_rel_dis = (2 * bucket_size - 1) * (2 * bucket_size - 1) + 3
            self.rel_pos_table = nn.Embedding(num_rel_dis, attention_heads)
            self.register_buffer("rp_bucket", make_image_bucket_position(bucket_size, num_rel_dis))

    def get_rel_pos_bias(self) -> Tensor:
        """Relative position bias, shape (H, L, L)."""
        return F.embedding(self.rp_bucket, self.rel_pos_table.weight).permute(2, 0, 1)

    def forward(self, src_images: Tensor) -> tuple[Tensor, Tensor | None]:
        """src_images: (B*T, 3, H, W) -> token (B*T, 1 + N, C), bias (1, heads, 1 + N, 1 + N)."""
        batch_size = src_images.size(0)
        expected = self.bucket_size * 16
        assert src_images.size(2) == expected, f"backbone expects {expected}x{expected} images"

        x = self.embed_images(src_images).flatten(2).transpose(1, 2)  # (BT) x N x C
        cls_embedding = self.cls_embedding.expand(batch_size, -1, -1)
        x = torch.cat([cls_embedding, x], dim=1)
        x = self.dropout_module(x + self.pos_embed.unsqueeze(0))

        # per-frame temporal embedding: (b t) n d + (1 t 1 d)
        n = x.shape[1]
        x = x.view(-1, self.num_frames, n, x.shape[-1]) + self.temporal_embedding.unsqueeze(2)
        x = x.view(batch_size, n, -1)

        attn_bias = self.get_rel_pos_bias().unsqueeze(0) if self.shared_rp_bias else None
        return x, attn_bias


class MultiheadAttention(nn.Module):
    """ONE-PEACE self-attention (LayerNorm before out_proj). Batch-first input (B, L, C)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, use_sdpa: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.use_sdpa = use_sdpa

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.ln = nn.LayerNorm(embed_dim)

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        """x: (B, L, C); attn_bias: None or broadcastable to (B, H, L, L)."""
        bsz, tgt_len, _ = x.shape
        shape = (bsz, tgt_len, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)  # B x H x L x d
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)

        if self.use_sdpa and not self.training:
            # softmax(q k^T / sqrt(d) + bias) v, identical to the matmul branch below
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            attn_weights = (q * self.scaling) @ k.transpose(-2, -1)
            if attn_bias is not None:
                attn_weights = attn_weights + attn_bias
            attn_probs = self.dropout_module(F.softmax(attn_weights, dim=-1))
            attn = attn_probs @ v

        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return self.out_proj(self.ln(attn))


class TransformerEncoderLayer(nn.Module):
    """Layer ONE-PEACE + adapter AIM: temporal attention -> spatial attention -> FFN (GeGLU)."""

    def __init__(
        self,
        attention_heads: int = 24,
        bucket_size: int = 16,
        dropout: float = 0.0,
        embed_dim: int = 1536,
        ffn_embed_dim: int = 6144,
        layer_scale_init_value: float = 1e-2,
        num_tadapter: int = 1,
        num_frames: int = 16,
        scale: float = 0.5,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.heads = attention_heads
        self.bucket_size = bucket_size
        self.self_attn = MultiheadAttention(embed_dim, attention_heads, use_sdpa=use_sdpa)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout_module = nn.Dropout(dropout)

        self.image_ffn = nn.Sequential(
            GeGLU(embed_dim, ffn_embed_dim),
            nn.Dropout(0.0),
            nn.LayerNorm(ffn_embed_dim),
            nn.Linear(ffn_embed_dim, embed_dim),
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones(embed_dim))
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones(embed_dim))

        self.MLP_Adapter = Adapter(embed_dim, skip_connect=False)
        self.S_Adapter = Adapter(embed_dim)
        self.scale = scale
        self.T_Adapter = Adapter(embed_dim, skip_connect=False)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(embed_dim)
        self.num_tadapter = num_tadapter
        self.num_frames = num_frames

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        """x: (B*T, N, C)."""
        n = x.shape[1]
        residual = x

        # temporal adaptation: attention across the T frames at the same token position
        xt = rearrange(x, "(b t) n d -> (b n) t d", t=self.num_frames)
        xt = self.self_attn_layer_norm(xt)
        if self.num_tadapter == 2:
            xt = self.T_Adapter_in(xt)
        xt = self.T_Adapter(self.self_attn(xt))
        x = x + rearrange(xt, "(b n) t d -> (b t) n d", n=n)

        # spatial adaptation: attention across the N tokens of one frame
        x = self.S_Adapter(self.self_attn(self.self_attn_layer_norm(x), attn_bias))
        x = residual + self.gamma_1 * x

        # joint adaptation
        residual = x
        xn = self.final_layer_norm(x)
        return (
            residual
            + self.gamma_2 * self.dropout_module(self.image_ffn(xn))
            + self.scale * self.MLP_Adapter(xn)
        )


def _prepare_attn_bias(bias: Tensor, dtype: torch.dtype) -> Tensor:
    """Lay out the bias (1, H, L, L) the way SDPA's memory-efficient kernel accepts it.

    The kernel needs the mask in q's dtype, a contiguous last dim and other strides divisible
    by 8; otherwise SDPA silently falls back to the slow fp32 "math" path. Values are
    unchanged: a wider buffer is allocated and sliced back to L columns.
    """
    length = bias.shape[-1]
    padded = bias.new_zeros(*bias.shape[:-1], -(-length // 8) * 8, dtype=dtype)
    padded[..., :length] = bias
    return padded[..., :length]


class TransformerEncoder(nn.Module):
    def __init__(self, layers: int = 40, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(TransformerEncoderLayer(**layer_kwargs) for _ in range(layers))
        self.num_layers = len(self.layers)
        self.image_layer_norm = nn.LayerNorm(layer_kwargs.get("embed_dim", 1536))

    def forward(self, image_info: tuple[Tensor, Tensor | None], cls_only: bool = False) -> Tensor:
        """Tokens (B*T, N, C), or only the CLS token (B*T, C) if ``cls_only``."""
        x, attn_bias = image_info
        if attn_bias is not None:
            attn_bias = _prepare_attn_bias(attn_bias, x.dtype)
        for layer in self.layers:
            x = layer(x, attn_bias)
        if cls_only:
            x = x[:, 0]  # LayerNorm is per token, so normalising CLS alone is enough
        return self.image_layer_norm(x)


class OnePeaceViT(nn.Module):
    """Defaults = configs/_base_/models/onepeace.py + configs/recognition/onepeace_k400.py."""

    def __init__(
        self,
        attention_heads: int = 24,
        adapter_scale: float = 0.5,
        bucket_size: int = 16,
        num_tadapter: int = 1,
        num_frames: int = 16,
        embed_dim: int = 1536,
        ffn_embed_dim: int = 6144,
        layers: int = 40,
        layer_scale_init_value: float = 1e-2,
        shared_rp_bias: bool = True,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.image_adapter = ImageAdaptor(
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            num_frames=num_frames,
            embed_dim=embed_dim,
            shared_rp_bias=shared_rp_bias,
        )
        self.encoder = TransformerEncoder(
            layers=layers,
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            layer_scale_init_value=layer_scale_init_value,
            num_tadapter=num_tadapter,
            num_frames=num_frames,
            scale=adapter_scale,
            use_sdpa=use_sdpa,
        )

    def set_use_sdpa(self, enabled: bool) -> None:
        for m in self.modules():
            if isinstance(m, MultiheadAttention):
                m.use_sdpa = enabled

    def forward(self, x: Tensor) -> Tensor:
        """x: normalised (B, C, T, H, W) -> per-frame CLS tokens (B, D, T)."""
        b, _, t = x.shape[:3]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.encoder(self.image_adapter(x), cls_only=True)  # (BT) x D
        return rearrange(x, "(b t) d -> b d t", b=b, t=t)

    @torch.no_grad()
    def extract_clip_features(self, x: Tensor) -> Tensor:
        """1536-d clip feature = CLS tokens averaged over time.

        This is exactly the input of I3DHead (spatial_type='avg') in K400 fine-tuning.
        """
        return self.forward(x).mean(dim=-1)


def load_k400_checkpoint(
    checkpoint_path: str, use_sdpa: bool = True, with_head: bool = False
) -> OnePeaceViT | tuple[OnePeaceViT, nn.Linear]:
    """Load ``onepeace_video_k400.pth`` (dict ``{'state_dict': {'backbone.*', 'cls_head.*'}}``).

    Returns the model (fp32, CPU, eval) and, with ``with_head``, the K400 Linear 1536 -> 400
    (only for a quick preprocessing sanity check).
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    except (TypeError, RuntimeError):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    prefix = "backbone."
    backbone_sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    num_frames = backbone_sd["image_adapter.temporal_embedding"].shape[1]
    try:
        # build on 'meta' (no RAM allocated) and assign the checkpoint tensors directly
        with torch.device("meta"):
            model = OnePeaceViT(num_frames=num_frames, use_sdpa=use_sdpa)
        model.load_state_dict(backbone_sd, strict=True, assign=True)
    except (TypeError, AttributeError):  # torch < 2.1
        model = OnePeaceViT(num_frames=num_frames, use_sdpa=use_sdpa)
        model.load_state_dict(backbone_sd, strict=True)
    model.eval()

    if not with_head:
        return model
    head_w, head_b = state_dict["cls_head.fc_cls.weight"], state_dict["cls_head.fc_cls.bias"]
    head = nn.Linear(head_w.shape[1], head_w.shape[0])
    with torch.no_grad():
        head.weight.copy_(head_w)
        head.bias.copy_(head_b)
    return model, head.eval()
