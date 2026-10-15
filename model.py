"""Event segmentation + captioning model, with no classification layer.

Backbone (from UniAV): one stream per modality (video V, audio A), conv embedding -> self-attention ->
cross-modal attention -> a pyramid of 6 levels (256, 128, 64, 32, 16, 8 steps).

Heads, on every pyramid level:
  - event head     (B, T, 1) : is this time step inside an event
  - boundary head  (B, T, 2) : distances to the two boundaries of that event,
                               plus an IoU-quality score used for ranking
  - embedding head (B, T, D) : content vector in the same space as the captions
  - ground head    (Q, T)    : given a sentence, is this time step inside the event the sentence describes

Inference: segments come from the event and boundary heads. Each segment's vector is the mean embedding
over its whole span (plus video context), compared by cosine with the caption pool to pick a caption.

The embedding losses run over the WHOLE train caption pool, because inference must pick from that pool.
"""

# in_dim=1024
# feat_dim=512
# clip_dim=512

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import config as C
from dataset import make_points


# =============================================================================================================
# building blocks
# =============================================================================================================

class MaskedConv1D(nn.Module):
    """Conv1d whose output is zeroed on padding steps; with stride > 1 the mask is downsampled too."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, kernel_size // 2,
                              groups=groups, bias=bias)
        if bias:
            nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        out = self.conv(x)
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            mask = F.interpolate(mask.float(), size=x.shape[-1] // self.stride, mode='nearest')
        out = out * mask.float()
        return out, mask.bool()


class LayerNorm(nn.Module):
    """LayerNorm over the channels of a (B, C, T) tensor."""

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones([1, num_channels, 1]))
        self.bias = nn.Parameter(torch.zeros([1, num_channels, 1]))

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        res = x - mu
        sigma = (res ** 2).mean(dim=1, keepdim=True)
        return res / torch.sqrt(sigma + self.eps) * self.weight + self.bias


def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table, a tensor of size (1, C, T)."""
    pos = np.arange(n_position)[:, None]
    hid = np.arange(d_hid)[None]
    table = pos / np.power(10000, 2 * (hid // 2) / d_hid)
    table[:, 0::2] = np.sin(table[:, 0::2])  # dim 2i
    table[:, 1::2] = np.cos(table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(table).unsqueeze(0).transpose(1, 2)


class MaskedMHCA(nn.Module):
    """Multi-head attention with a depthwise conv on queries, keys and values (ActionFormer).
    The conv encodes local order and, with stride 2, halves the length (the pyramid levels).
    Queries come from x2, keys and values from x1 (x1 is x2 for self-attention)."""

    def __init__(self, n_embd, n_head, stride=1):
        super().__init__()
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        kernel_size = stride + 1 if stride > 1 else 3
        # 1d depthwise convs
        self.query_conv = MaskedConv1D(n_embd, n_embd, kernel_size, stride=stride, groups=n_embd, bias=False)
        self.query_norm = LayerNorm(n_embd)
        self.key_conv = MaskedConv1D(n_embd, n_embd, kernel_size, stride=stride, groups=n_embd, bias=False)
        self.key_norm = LayerNorm(n_embd)
        self.value_conv = MaskedConv1D(n_embd, n_embd, kernel_size, stride=stride, groups=n_embd, bias=False)
        self.value_norm = LayerNorm(n_embd)

        # key, query, value projections for all heads
        self.key = nn.Conv1d(n_embd, n_embd, 1)
        self.query = nn.Conv1d(n_embd, n_embd, 1)
        self.value = nn.Conv1d(n_embd, n_embd, 1)
        # output projection
        self.proj = nn.Conv1d(n_embd, n_embd, 1)

    def forward(self, x1, x2, mask):
        B, Cn, T = x1.size()
        q, q_mask = self.query_conv(x2, mask)
        k, kv_mask = self.key_conv(x1, mask)
        v, _ = self.value_conv(x1, mask)
        q = self.query(self.query_norm(q))
        k = self.key(self.key_norm(k))
        v = self.value(self.value_norm(v))

        # (B, nh * hs, T') -> (B, nh, T', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # (B, nh, T', hs) x (B, nh, hs, T') -> (B, nh, T', T')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to padding steps
        att = att.masked_fill(~kv_mask[:, :, None, :], float('-inf'))
        att = F.softmax(att, dim=-1)
        # (B, nh, T', T') x (B, nh, T', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].float())
        out = out.transpose(2, 3).contiguous().view(B, Cn, -1)
        out = self.proj(out) * q_mask.float()
        return out, q_mask


def drop_path(x, drop_prob, training):
    """Stochastic depth: drop the whole residual branch of a sample with probability drop_prob."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    mask = (keep_prob + torch.rand((x.shape[0], 1, 1), dtype=x.dtype, device=x.device)).floor()
    return x / keep_prob * mask


class AffineDropPath(nn.Module):
    """Drop path with a per-channel scale that starts at 1e-4 (https://arxiv.org/pdf/2103.17239.pdf)."""

    def __init__(self, num_dim, drop_prob):
        super().__init__()
        self.scale = nn.Parameter(1e-4 * torch.ones((1, num_dim, 1)))
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)


class TransformerBlock(nn.Module):
    """out = x1 + Attn(q=x2, k=x1, v=x1), then an MLP (pre-LN). With stride 2 the output is half as long."""

    def __init__(self, n_embd, n_head, stride=1, path_pdrop=0.0):
        super().__init__()
        self.ln11 = LayerNorm(n_embd)
        self.ln12 = LayerNorm(n_embd)
        self.attn = MaskedMHCA(n_embd, n_head, stride)
        if stride > 1:
            self.pool_skip = nn.MaxPool1d(stride + 1, stride=stride, padding=(stride + 1) // 2)
        else:
            self.pool_skip = nn.Identity()
        self.ln2 = LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, 4 * n_embd, 1),
            nn.GELU(),
            nn.Conv1d(4 * n_embd, n_embd, 1),
        )
        self.drop_path_attn = AffineDropPath(n_embd, path_pdrop)
        self.drop_path_mlp = AffineDropPath(n_embd, path_pdrop)

    def forward(self, x1, x2, mask):
        out, out_mask = self.attn(self.ln11(x1), self.ln12(x2), mask)
        m = out_mask.float()
        out = self.pool_skip(x1) * m + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * m)
        return out, out_mask


class Scale(nn.Module):
    """Multiply the input by a learnable scalar."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


# =============================================================================================================
# backbone
# =============================================================================================================

class Backbone(nn.Module):
    """
        Visual (V) and audio (A) features each go through: conv embedding -> 2 self-attention blocks
        -> one cross-modal block at full resolution -> 5 cross-modal blocks that each halve the length.
        Returns the 6 pyramid levels of each stream, and their masks.

        Cross-modal block of stream X with the other stream Y: out = X + Attention(query = Y, key = X, value = X),
        i.e. X is re-weighted over time by what Y finds relevant.
    """

    def __init__(self):
        super().__init__()
        n = C.EMBD_DIM
        self.register_buffer('pos_embd', get_sinusoid_encoding(C.MAX_SEQ_LEN, n) / (n ** 0.5), persistent=False)

        # embedding network using convs
        # 768 -> 512
        self.embd_V = nn.ModuleList()
        self.embd_A = nn.ModuleList()
        self.embd_norm_V = nn.ModuleList()
        self.embd_norm_A = nn.ModuleList()
        for i in range(C.ARCH[0]):
            self.embd_V.append(MaskedConv1D(C.INPUT_DIM_V if i == 0 else n, n, 3, bias=False))
            self.embd_A.append(MaskedConv1D(C.INPUT_DIM_A if i == 0 else n, n, 3, bias=False))
            self.embd_norm_V.append(LayerNorm(n))
            self.embd_norm_A.append(LayerNorm(n))

        # stem: self-attention
        self.self_att_V = nn.ModuleList()
        self.self_att_A = nn.ModuleList()
        for i in range(C.ARCH[1] - 1):
            self.self_att_V.append(TransformerBlock(n, C.N_HEAD, 1, C.DROPPATH))
            self.self_att_A.append(TransformerBlock(n, C.N_HEAD, 1, C.DROPPATH))

        #cross-attention on original temporal resolution
        self.ori_cross_att_Va = TransformerBlock(n, C.N_HEAD, 1, C.DROPPATH)
        self.ori_cross_att_Av = TransformerBlock(n, C.N_HEAD, 1, C.DROPPATH)

        #cross-attention after down-sampling
        self.cross_att_Va = nn.ModuleList()
        self.cross_att_Av = nn.ModuleList()
        for i in range(C.ARCH[2]):
            self.cross_att_Va.append(TransformerBlock(n, C.N_HEAD, C.SCALE_FACTOR, C.DROPPATH))
            self.cross_att_Av.append(TransformerBlock(n, C.N_HEAD, C.SCALE_FACTOR, C.DROPPATH))

        # set nn.Linear/nn.Conv1d bias term to 0
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x_V, x_A, mask):
        # x_V/x_A: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        T = x_V.shape[-1]
        mask_V = mask_A = mask
        # embedding network
        for i in range(len(self.embd_V)):
            x_V, mask_V = self.embd_V[i](x_V, mask_V)
            x_V = F.relu(self.embd_norm_V[i](x_V))
            x_A, mask_A = self.embd_A[i](x_A, mask_A)
            x_A = F.relu(self.embd_norm_A[i](x_A))

        # position embedding
        x_V = x_V + self.pos_embd[:, :, :T] * mask_V.float()
        x_A = x_A + self.pos_embd[:, :, :T] * mask_A.float()

        # stem transformer
        for i in range(len(self.self_att_V)):
            x_V, mask_V = self.self_att_V[i](x_V, x_V, mask_V)
            x_A, mask_A = self.self_att_A[i](x_A, x_A, mask_A)

        # level 0: x_Va = V updated with audio queries, x_Av = A updated with visual queries
        x_Va, mask_V = self.ori_cross_att_Va(x_V, x_A, mask_V)
        x_Av, mask_A = self.ori_cross_att_Av(x_A, x_V, mask_A)

        feats_V, feats_A, masks = [x_Va], [x_Av], [mask_V]
        # level i + 1 is built from level i of both streams
        for i in range(len(self.cross_att_Va)):
            x_V, mask_V = self.cross_att_Va[i](feats_V[i], feats_A[i], mask_V)
            x_A, mask_A = self.cross_att_Av[i](feats_A[i], feats_V[i], mask_A)
            feats_V.append(x_V)
            feats_A.append(x_A)
            masks.append(mask_V)
        return feats_V, feats_A, masks


# =============================================================================================================
# heads
# =============================================================================================================

def conv_stack(in_dim, feat_dim): # not change T
    """The shared body of every head: 2 masked Conv1D layers, each followed by a LayerNorm."""
    head = nn.ModuleList([MaskedConv1D(in_dim, feat_dim, 3, bias=False),
                          MaskedConv1D(feat_dim, feat_dim, 3, bias=False)])
    norm = nn.ModuleList([LayerNorm(feat_dim), LayerNorm(feat_dim)])
    return head, norm


def run_stack(head, norm, x, m):
    for i in range(len(head)):
        x, _ = head[i](x, m)
        x = F.relu(norm[i](x))
    return x


class EventHead(nn.Module):
    """
    Iterate for each level of T = [256, 128, 64, 32, 16, 8], and check if each level has event occured.

    Returns:
        Tuple[tensor[B, T, 1]]
    """

    def __init__(self, in_dim, feat_dim, prior_prob=0.01):
        super().__init__()
        self.head, self.norm = conv_stack(in_dim, feat_dim)
        self.out = MaskedConv1D(feat_dim, 1, 3)
        nn.init.constant_(self.out.conv.bias, -np.log((1 - prior_prob) / prior_prob))

    def forward(self, fpn_feats, fpn_masks):
        res = tuple()
        for x, m in zip(fpn_feats, fpn_masks):
            x = run_stack(self.head, self.norm, x, m)
            o, _ = self.out(x, m)
            res += (o.permute(0, 2, 1),)          # (B, T_l, 1)
        return res # tuple[tensor(B, T_l, 1)]


class EmbedHead(nn.Module):
    """One L2-normalised vector per time step, in the space of the projected captions."""

    def __init__(self, in_dim, feat_dim):
        super().__init__()
        self.head, self.norm = conv_stack(in_dim, feat_dim)
        self.vis_proj = MaskedConv1D(feat_dim, feat_dim, 3)
        self.clip_proj = nn.Linear(C.CLIP_DIM, feat_dim) # 512 -> 512
        nn.init.xavier_uniform_(self.clip_proj.weight)
        nn.init.constant_(self.clip_proj.bias, 0.0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, fpn_feats, fpn_masks):
        """Return the normalised vectors of every time step, plus the raw features of
        level 0 (finest resolution) for span averaging."""
        res, raw0 = tuple(), None
        for x, m in zip(fpn_feats, fpn_masks):
            x = run_stack(self.head, self.norm, x, m)
            x, _ = self.vis_proj(x, m)
            if raw0 is None:
                raw0 = x                                       # (B, D, T0)
            res += (F.normalize(x, dim=1).permute(0, 2, 1),)   # (B, T, D)
        return res, raw0

    def embed_captions(self, cap):
        return F.normalize(self.clip_proj(cap), dim=-1)


# raw (512, 256) level 0, not normalized

def span_mean(raw, length, segs):
    """Mean of raw features (D, T) over spans segs (N, 2), in grid units.

    Uses a cumulative sum, so there is no loop over segments. A span shorter than
    one step still covers at least one step.
    """
    cs = F.pad(raw.cumsum(-1), (1, 0))                         # (D, T+1)
    lo = segs[:, 0].floor().clamp(0, length - 1).long()
    hi = segs[:, 1].ceil().long() + 1
    hi = torch.maximum(hi.clamp(max=length), lo + 1)
    return (cs[:, hi] - cs[:, lo]).t() / (hi - lo).unsqueeze(1).float()   # (N, D)


class SegmentContext(nn.Module):
    """Segment vector with context from the whole video.

    Inputs are the span mean, the whole-video mean (which dish is being cooked) and
    the relative position of the span (early or late recipe step). The last layer
    starts at zero, so initially this is exactly the span mean.
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + 2, dim), # span + glob + pos = 512 + 512 + 2
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, raw, length, segs):
        span = span_mean(raw, length, segs)                          # (N, D)
        glob = raw[:, :length].mean(-1).expand_as(span)              # (N, D)
        pos = (segs / float(length)).clamp(0, 1)                     # (N, 2)
        return F.normalize(span + self.mlp(torch.cat((span, glob, pos), -1)), dim=-1)


class BoundaryHead(nn.Module):
    """Per time step: distances to the start and the end of its event (in units of the level's stride,
    >= 0) and one IoU-quality logit (how good the segment from this step will be, used for ranking)."""

    def __init__(self, in_dim, feat_dim):
        super().__init__()
        self.head, self.norm = conv_stack(in_dim, feat_dim)
        self.scale = nn.ModuleList([Scale() for _ in C.FPN_STRIDES])
        self.out = MaskedConv1D(feat_dim, 2, 3)
        # a positive bias starts every step inside the ReLU's active range (offsets are in stride units)
        nn.init.constant_(self.out.conv.bias, 1.0)
        # predicts the IoU between the segment from this step and the GT, used for ranking
        self.iou_out = MaskedConv1D(feat_dim, 1, 3)

    def forward(self, fpn_feats, fpn_masks):
        res, qual = tuple(), tuple()
        for l, (x, m) in enumerate(zip(fpn_feats, fpn_masks)):
            x = run_stack(self.head, self.norm, x, m)
            q, _ = self.iou_out(x, m)
            qual += (q.permute(0, 2, 1),)                         # (B, T_l, 1)
            o, _ = self.out(x, m)
            res += (F.relu(self.scale[l](o)).permute(0, 2, 1),)  # (B, T_l, 2)
        return res, qual


class GroundHead(nn.Module):
    """Query-conditioned event head: is this time step inside the event the sentence describes.

    The query (a caption vector) modulates the features after the first conv layer (FiLM: a per-channel
    scale and shift, both starting at zero), and the cosine between each step's embedding and the projected
    query is added to the logit, so the head starts from what the embedding head already knows.
    """

    def __init__(self, in_dim, feat_dim, text_dim, prior_prob=0.01):
        super().__init__()
        self.head, self.norm = conv_stack(in_dim, feat_dim)
        #                               gamma x beta
        self.film = nn.Linear(text_dim, 2 * feat_dim) # FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., AAAI 2018)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.out = MaskedConv1D(feat_dim, 1, 3)
        nn.init.constant_(self.out.conv.bias, -np.log((1 - prior_prob) / prior_prob))
        self.cos_scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, fpn_feats, fpn_masks, vid, query, query_e, em):
        """vid (Q,) video of each query in the batch, query (Q, text_dim) caption vectors,
        query_e (Q, D) the same projected into the event space, em (B, P, D) step embeddings.
        Returns logits (Q, P) over the steps of every pyramid level."""
        gamma, beta = self.film(query).chunk(2, dim=-1)                      # (Q, C)
        res = []
        for x, m in zip(fpn_feats, fpn_masks):
            x, _ = self.head[0](x, m)
            x = F.relu(self.norm[0](x))[vid]                                 # (Q, C, T)
            m = m[vid]
            x = x * (1 + gamma[:, :, None]) + beta[:, :, None]
            x, _ = self.head[1](x, m)
            x = F.relu(self.norm[1](x))
            o, _ = self.out(x, m)
            res.append(o.squeeze(1))                                         # (Q, T)
        cos = (em[vid] * query_e[:, None]).sum(-1)                           # (Q, P)
        return torch.cat(res, dim=1) + self.cos_scale * cos


# =============================================================================================================
# losses and NMS
# =============================================================================================================

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Focal loss of RetinaNet (https://arxiv.org/abs/1708.02002), summed."""
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).sum()


def diou_loss_1d(pred, target):
    """Distance-IoU loss (https://arxiv.org/abs/1911.08287) of 1D segments given as (left, right) offsets
    from the same point, summed."""
    lp, rp = pred[:, 0], pred[:, 1]
    lg, rg = target[:, 0], target[:, 1]
    inter = torch.min(lp, lg) + torch.min(rp, rg)
    union = (lp + rp) + (lg + rg) - inter
    iou = inter / union.clamp(min=1e-8)
    # smallest enclosing segment, and the distance between the two centres
    len_c = torch.max(lp, lg) + torch.max(rp, rg)
    rho = 0.5 * (rp - lp - rg + lg)
    return (1.0 - iou + torch.square(rho / len_c.clamp(min=1e-8))).sum()


def soft_nms(segs, scores, sigma=0.5, min_score=0.001, max_num=100):
    """Gaussian soft-NMS on 1D segments: repeatedly take the best segment and lower the score of every other
    segment by exp(-IoU^2 / sigma). Returns the kept segments, scores and their indices, best first."""
    idx = torch.arange(len(segs))
    keep_segs, keep_scores, keep_idx = [], [], []
    while len(scores) > 0 and len(keep_idx) < max_num:
        i = scores.argmax()
        s, sc = segs[i], scores[i]
        keep_segs.append(s); keep_scores.append(sc); keep_idx.append(idx[i])
        rest = torch.arange(len(scores)) != i
        segs, scores, idx = segs[rest], scores[rest], idx[rest]
        inter = (torch.min(segs[:, 1], s[1]) - torch.max(segs[:, 0], s[0])).clamp(min=0)
        union = (s[1] - s[0] + 1e-6) + (segs[:, 1] - segs[:, 0] + 1e-6) - inter
        scores = scores * torch.exp(-(inter / union) ** 2 / sigma)
        ok = scores >= min_score
        segs, scores, idx = segs[ok], scores[ok], idx[ok]
    if not keep_idx:
        return segs, scores, idx
    return torch.stack(keep_segs), torch.stack(keep_scores), torch.stack(keep_idx)


# =============================================================================================================
# the model
# =============================================================================================================

class EventCaptionModel(nn.Module):
    """Segment events, then caption them. No classification layer."""

    def __init__(self, pool_emb):
        """pool_emb (N, 512): the train caption pool."""
        super().__init__()
        self.backbone = Backbone()
        D = C.EMBD_DIM * 2 # 512 * 2
        self.event_head = EventHead(D, C.HEAD_DIM)
        self.bound_head = BoundaryHead(D, C.HEAD_DIM)
        self.embed_head = EmbedHead(D, C.HEAD_DIM)
        self.seg_ctx = SegmentContext(C.HEAD_DIM)
        self.ground_head = GroundHead(D, C.HEAD_DIM, C.CLIP_DIM)

        self.register_buffer('points', make_points(), persistent=False)        # (P, 4)
        self.register_buffer('pool_raw', pool_emb, persistent=False)

    def preprocessing(self, batch):
        """Stack the videos of a batch into (B, C=768, T=256) tensors plus a (B, 1, T) validity mask.
        A cropped training video is shorter than 256 steps and is padded with zeros."""
        dev = self.points.device
        fv = [x['visual'] for x in batch] # x['visual'] = (C, T)
        fa = [x['audio'] for x in batch]
        lens = torch.as_tensor([f.shape[-1] for f in fv]) # T are different because of data agumnetation

        def pad(xs):
            """Padding function"""
            # new_full: create tensor same device and dtype
            #                        B           C           T
            out = xs[0].new_full([len(xs), xs[0].shape[0], C.MAX_SEQ_LEN], 0.0)
            for s, d in zip(xs, out):
                d[..., :s.shape[-1]].copy_(s)
            return out.to(dev)

        masks = (torch.arange(C.MAX_SEQ_LEN)[None, :] < lens[:, None]).unsqueeze(1).to(dev) # (1, max_len) < (N, 1) = (N, max_len) -> (N, 1, max_len)
        return pad(fv), pad(fa), masks # (B, C, T)

    def forward(self, batch):
        """Training: dict of losses (final_loss is the weighted sum). Evaluation: one dict per video.
        P below is the number of points over all pyramid levels (504 for 256 steps): the outputs of
        every level are concatenated along the time axis."""

        V, A, masks = self.preprocessing(batch) # V = A = (B, C, T)

        # RETURN PYRIMID FEATURE
        fV, fA, msk = self.backbone(V, A, masks) # fV = fA = (B, C, T_l), l ∈ [0, 5]

        feats = [torch.cat((v, a), dim=1) for v, a in zip(fV, fA)]           # per level (B, 2 * embd_dim[512], T_l), len(feats) = 6

        # Event Head
        #                   tuple(tensor(B, T_l, 1))
        ev = torch.cat(self.event_head(feats, msk), dim=1).squeeze(-1)   # (B, P), P = point = 504

        # Bounding Head
        # tuple(tensor(B, T_l, 2)), tuple(tensor(B, T_l, 1))
        bd, qu = self.bound_head(feats, msk)
        bd = torch.cat(bd, dim=1)                                        # (B, P, 2)
        qu = torch.cat(qu, dim=1).squeeze(-1)                            # (B, P)

        # Embedding Head
        em, raw0 = self.embed_head(feats, msk)
        em = torch.cat(em, dim=1)                                        # (B, P, D)
        len0 = msk[0].squeeze(1).sum(-1)                                 # (B,)
        valid = torch.cat([m.squeeze(1) for m in msk], dim=1)            # (B, P)
        self.fpn = (feats, msk, em)                                      # for the ground head
        if self.training:
            return self.losses(batch, valid, ev, bd, qu, em, raw0, len0)
        return self.inference(batch, valid, ev, bd, qu, em, raw0, len0)

    def losses(self, batch, valid, ev, bd, qu, em, raw0, len0):
        dev = ev.device
        gt_cls = torch.stack([x['gt_cls'] for x in batch]).to(dev)       # (B, P, NMAX)
        gt_off = torch.stack([x['gt_off'] for x in batch]).to(dev)       # (B, P, 2)
        cap_pool = torch.stack([x['cap_pool_idx'] for x in batch]).to(dev)   # (B, NMAX)

        is_event = gt_cls.sum(-1) > 0                                    # (B, P)
        pos = is_event & valid
        n_pos = max(int(pos.sum()), 1)

        # 1. event or not: binary, no classes
        ev_loss = sigmoid_focal_loss(ev[valid], is_event[valid].float()) / n_pos

        # 2. segment boundaries, on the points inside an event
        rg_loss = diou_loss_1d(bd[pos], gt_off[pos]) / n_pos

        # 3. boundary quality: predict the IoU between the predicted and the GT segment, used for ranking
        p, g = bd[pos].detach(), gt_off[pos]
        inter = torch.min(p[:, 0], g[:, 0]) + torch.min(p[:, 1], g[:, 1])
        union = p.sum(-1) + g.sum(-1) - inter
        iou_loss = F.binary_cross_entropy_with_logits(qu[pos], inter / union)

        # 4. per-step embedding: each point inside an event must pick its caption out of the whole pool
        pool_f = self.embed_head.embed_captions(self.pool_raw)          # (N, D)
        scale = self.embed_head.logit_scale.exp().clamp(max=100)
        bi, pi = pos.nonzero(as_tuple=True)
        gt = cap_pool[bi, gt_cls[bi, pi].argmax(-1)]                     # pool index of each point's caption
        emb_loss = F.cross_entropy(scale * em[bi, pi] @ pool_f.t(), gt)

        # 5. span embedding: the vector of each GT span must pick its caption out of the whole pool
        qs, gs = [], []
        for b, x in enumerate(batch):
            qs.append(self.seg_ctx(raw0[b], int(len0[b]), x['segments'].to(dev).float()))
            gs.append(cap_pool[b, x['labels'].to(dev)])
        span_loss = F.cross_entropy(scale * torch.cat(qs) @ pool_f.t(), torch.cat(gs))

        # 6. grounding: each caption of a video, used as a query, must find the points of its own step
        vid, query, tgt = [], [], []
        for b, x in enumerate(batch):
            for j in x['labels'].tolist():
                vid.append(b)
                query.append(self.pool_raw[cap_pool[b, j]])
                tgt.append(gt_cls[b, :, j])
        vid = torch.as_tensor(vid, device=dev)
        logit = self.ground(vid, torch.stack(query))                    # (Q, P)
        tgt = torch.stack(tgt)
        ok = valid[vid]
        ground_loss = sigmoid_focal_loss(logit[ok], tgt[ok]) / tgt[ok].sum().clamp(min=1)

        total = (ev_loss + C.W_REG * rg_loss + C.W_IOU * iou_loss + C.W_EMB * emb_loss + C.W_SPAN * span_loss
                 + C.W_GROUND * ground_loss)
        return {'ev_loss': ev_loss, 'reg_loss': rg_loss, 'iou_loss': iou_loss, 'emb_loss': emb_loss,
                'span_loss': span_loss, 'ground_loss': ground_loss, 'final_loss': total}

    def ground(self, vid, query):
        """Ground-head logits (Q, P) on the last forward pass; vid (Q,) batch index of each query,
        query (Q, 512) caption vectors."""
        feats, msk, em = self.fpn
        return self.ground_head(feats, msk, vid, query, self.embed_head.embed_captions(query), em)

    def to_seconds(self, segs, x):
        """Grid positions -> seconds (step i is centred on (i + 0.5) * step_sec)."""
        return ((segs + 0.5) * x['step_sec']).clamp(min=0.0, max=float(x['duration']))

    @torch.no_grad()
    def inference(self, batch, valid, ev, bd, qu, em, raw0, len0):
        """Segment with event score times boundary-quality score, and return the segment vectors.

        One dict per video: segments (K, 2) in seconds, best first; embeds (K, D) segment vectors;
        ground_segments the best segment of every GT caption used as a
        query; gt_segments, gt_text the GT steps in seconds and their captions.
        """
        a = C.IOU_POWER
        prob = ev.sigmoid().pow(1 - a) * qu.sigmoid().pow(a) * valid.float()
        pts = self.points
        out = []
        for b, x in enumerate(batch):
            keep = (prob[b] > C.SCORE_THRESH).nonzero(as_tuple=True)[0]
            if len(keep) == 0:
                keep = prob[b].topk(1).indices
            sc, order = prob[b][keep].topk(min(C.MAX_SEG * 5, len(keep)))
            keep = keep[order]
            # segment of a point: from t - (start offset) * stride to t + (end offset) * stride
            p, o = pts[keep], bd[b][keep]
            segs = torch.stack((p[:, 0] - o[:, 0] * p[:, 3], p[:, 0] + o[:, 1] * p[:, 3]), -1).cpu()
            scores = sc.cpu()
            ok = (segs[:, 1] - segs[:, 0]) > C.DURATION_THRESH
            segs, scores, _ = soft_nms(segs[ok], scores[ok], C.NMS_SIGMA, 0.001, C.MAX_SEG)
            # segment vector = mean over the whole span (in grid units)
            span = self.seg_ctx(raw0[b], int(len0[b]), segs.to(raw0.device))

            # each GT caption as a query for the ground head: its best segment
            n = len(x['segments_sec'])
            gl = self.ground(torch.full((n,), b, device=ev.device), x['cap_emb'][:n].to(ev.device))
            gp = gl.sigmoid().pow(1 - a) * qu[b][None].sigmoid().pow(a) * valid[b][None].float()
            best = gp.argmax(-1)
            gpt, go = pts[best], bd[b][best]
            gr = torch.stack((gpt[:, 0] - go[:, 0] * gpt[:, 3], gpt[:, 0] + go[:, 1] * gpt[:, 3]), -1).cpu()

            out.append({'video_id': x['video_id'], 'segments': self.to_seconds(segs, x), 'scores': scores,
                        'embeds': span.cpu(),
                        'ground_segments': self.to_seconds(gr, x),
                        'gt_segments': x['segments_sec'], 'gt_text': x['cap_text']})
        return out
