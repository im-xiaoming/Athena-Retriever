"""Inference-only copy of libs/modeling/event_archs.py:EventCaptionTransformer.

Parameter names are unchanged, so training checkpoints (ckpt/<run>/best_cap.pth.tar) load
directly. Losses, GT handling and the teacher losses are removed; the optional
OmniRetriever projection is kept only so teacher-trained checkpoints load strictly.

Input : visual and audio features (C, T) already resampled to the grid (see pipeline.py).
Output: candidate segments in grid units, their scores, and one L2-normalised vector per
        segment in the caption space (the space captions are projected into by clip_proj).
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import ConvTransformerBackbone
from .blocks import MaskedConv1D, Scale, LayerNorm, Linear
from .point_generator import PointGenerator
from ..postprocess import soft_nms


def _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln):
    head, norm = nn.ModuleList(), nn.ModuleList()
    for i in range(n_layers - 1):
        head.append(MaskedConv1D(in_dim if i == 0 else feat_dim, feat_dim, ks,
                                 stride=1, padding=ks // 2, bias=(not with_ln)))
        norm.append(LayerNorm(feat_dim) if with_ln else nn.Identity())
    return head, norm


class EventHead(nn.Module):
    def __init__(self, in_dim, feat_dim, n_layers=3, ks=3, with_ln=True):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)

    def forward(self, fpn_feats, fpn_masks):
        res = tuple()
        for x, m in zip(fpn_feats, fpn_masks):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            o, _ = self.out(x, m)
            res += (o.permute(0, 2, 1),)
        return res


class EmbedHead(nn.Module):
    def __init__(self, in_dim, feat_dim, n_layers=3, ks=3, with_ln=True, clip_dim=1536):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.vis_proj = MaskedConv1D(feat_dim, feat_dim, ks, stride=1, padding=ks // 2)
        self.clip_proj = Linear(clip_dim, feat_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, fpn_feats, fpn_masks):
        """Normalised vectors of every step, plus level-0 raw features for span averaging."""
        res, raw0 = tuple(), None
        for x, m in zip(fpn_feats, fpn_masks):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            x, _ = self.vis_proj(x, m)
            if raw0 is None:
                raw0 = x
            res += (F.normalize(x, dim=1).permute(0, 2, 1),)
        return res, raw0

    def embed_captions(self, cap):
        """InternVideo2 caption or query vectors (N, 512) -> event space (N, D), normalised."""
        return F.normalize(self.clip_proj(cap), dim=-1)


def span_mean(raw, length, segs):
    """Mean of raw (D, T) over spans segs (N, 2) in grid units; at least one step per span."""
    cs = F.pad(raw.cumsum(-1), (1, 0))
    lo = segs[:, 0].floor().clamp(0, length - 1).long()
    hi = segs[:, 1].ceil().long() + 1
    hi = torch.maximum(hi.clamp(max=length), lo + 1)
    return (cs[:, hi] - cs[:, lo]).t() / (hi - lo).unsqueeze(1).float()


class SegmentContext(nn.Module):
    """Span mean + whole-video mean + relative position -> segment vector."""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim + 2, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, raw, length, segs):
        span = span_mean(raw, length, segs)
        glob = raw[:, :length].mean(-1).expand_as(span)
        pos = (segs / float(length)).clamp(0, 1)
        return F.normalize(span + self.mlp(torch.cat((span, glob, pos), -1)), dim=-1)


class BoundaryHead(nn.Module):
    def __init__(self, in_dim, feat_dim, fpn_levels, n_layers=3, ks=3, with_ln=True):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.scale = nn.ModuleList([Scale() for _ in range(fpn_levels)])
        self.out = MaskedConv1D(feat_dim, 2, ks, stride=1, padding=ks // 2)
        self.iou_out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)

    def forward(self, fpn_feats, fpn_masks):
        res, qual = tuple(), tuple()
        for l, (x, m) in enumerate(zip(fpn_feats, fpn_masks)):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            o, _ = self.out(x, m)
            q, _ = self.iou_out(x, m)
            res += (F.relu(self.scale[l](o)).permute(0, 2, 1),)
            qual += (q.permute(0, 2, 1),)
        return res, qual


class EventCaptionModel(nn.Module):
    """Segment events and give each one a caption-space vector."""

    def __init__(self, backbone_arch, scale_factor, input_dim_V, input_dim_A, n_head, embd_kernel_size,
                 embd_dim, embd_with_ln, head_dim, regression_range, head_num_layers, head_kernel_size,
                 head_with_ln, use_abs_pe, max_seq_len, test_cfg, train_cfg=None, clip_dim=1536, omni_dim=0,
                 **unused):
        super().__init__()
        self.fpn_strides = [scale_factor ** i for i in range(backbone_arch[-1] + 1)]
        self.max_seq_len = max_seq_len
        self.max_div_factor = max(self.fpn_strides)
        self.score_thresh = test_cfg['score_thresh']
        self.duration_thresh = test_cfg['duration_thresh']
        self.max_seg = test_cfg['max_seg_num']
        self.nms_sigma = test_cfg.get('nms_sigma', 0.5)
        self.iou_power = test_cfg.get('iou_power', 0.3)
        self.backbone = ConvTransformerBackbone(
            n_in_V=input_dim_V, n_in_A=input_dim_A, n_embd=embd_dim, n_head=n_head,
            n_embd_ks=embd_kernel_size, max_len=max_seq_len, arch=backbone_arch,
            scale_factor=scale_factor, with_ln=embd_with_ln, attn_pdrop=0.0, proj_pdrop=0.0,
            # droppath > 0 builds AffineDropPath, whose learned per-channel scale is still applied
            # in eval mode; with 0 the blocks become Identity and those weights would be lost
            path_pdrop=(train_cfg or {}).get('droppath', 0.1), use_abs_pe=use_abs_pe)
        D = embd_dim * 2
        self.event_head = EventHead(D, head_dim, head_num_layers, head_kernel_size, head_with_ln)
        self.bound_head = BoundaryHead(D, head_dim, len(self.fpn_strides), head_num_layers,
                                       head_kernel_size, head_with_ln)
        self.embed_head = EmbedHead(D, head_dim, head_num_layers, head_kernel_size, head_with_ln, clip_dim)
        self.seg_ctx = SegmentContext(head_dim)
        if omni_dim > 0:   # only so teacher-trained checkpoints load; unused at inference
            self.omni_proj = nn.Sequential(nn.Linear(head_dim, head_dim * 2), nn.GELU(),
                                           nn.Linear(head_dim * 2, omni_dim))
            self.omni_logit_scale = nn.Parameter(torch.ones([]))
        self.points = PointGenerator(max_seq_len, 4, len(self.fpn_strides), scale_factor,
                                     regression_range, self.max_div_factor)

    @torch.no_grad()
    def forward(self, visual, audio):
        """visual, audio: (C, T) on the model's device. Returns segments (grid units), scores, vectors."""
        dev = visual.device
        T = visual.shape[-1]
        max_len = self.max_seq_len if T <= self.max_seq_len else \
            (T + self.max_div_factor - 1) // self.max_div_factor * self.max_div_factor
        V = visual.new_zeros(1, visual.shape[0], max_len); V[0, :, :T] = visual
        A = audio.new_zeros(1, audio.shape[0], max_len); A[0, :, :T] = audio
        mask = (torch.arange(max_len, device=dev)[None] < T).unsqueeze(1)
        fV, fA, msk = self.backbone(V, A, mask)
        feats = [torch.cat((v, a), 1) for v, a in zip(fV, fA)]
        ev = torch.cat(self.event_head(feats, msk), dim=1).squeeze(-1)[0]
        bd, qu = self.bound_head(feats, msk)
        bd = torch.cat(bd, dim=1)[0]
        qu = torch.cat(qu, dim=1).squeeze(-1)[0]
        em, raw0 = self.embed_head(feats, msk)
        em = torch.cat(em, dim=1)[0]
        valid = torch.cat([m.squeeze(1) for m in msk], dim=1)[0]
        len0 = int(msk[0].squeeze(1).sum())

        a = self.iou_power
        prob = ev.sigmoid().pow(1 - a) * qu.sigmoid().pow(a) * valid.float()
        pts = torch.cat(self.points(self.fpn_strides, T), dim=0).to(dev)
        keep = (prob > self.score_thresh).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            keep = prob.topk(1).indices
        sc, order = prob[keep].topk(min(self.max_seg * 5, keep.numel()))
        keep = keep[order]
        p, o = pts[keep], bd[keep]
        segs = torch.stack((p[:, 0] - o[:, 0] * p[:, 3], p[:, 0] + o[:, 1] * p[:, 3]), -1)
        ok = (segs[:, 1] - segs[:, 0]) > self.duration_thresh
        segs, sc = segs[ok], sc[ok]
        segs_np, sc_np, _ = soft_nms(segs.float().cpu().numpy(), sc.float().cpu().numpy(),
                                     sigma=self.nms_sigma, min_score=0.001, max_num=self.max_seg)
        segs = torch.from_numpy(segs_np).to(dev)
        vecs = self.seg_ctx(raw0[0], len0, segs) if len(segs) else torch.zeros(0, raw0.shape[1], device=dev)
        self.last_level0 = (raw0[0], len0)   # for span_tokens (caption generator)
        return segs_np, sc_np, vecs

    @torch.no_grad()
    def span_tokens(self, segs, k=8):
        """Level-0 Embed Head features of the last forward() sampled at k points across each segment
        (grid units), (N, k, D): the generator input, as in tools/capgen/dump_segments.py."""
        raw0, L = self.last_level0
        segs = torch.as_tensor(segs, dtype=torch.float32, device=raw0.device).reshape(-1, 2)
        t = torch.linspace(0, 1, k, device=raw0.device)
        pos = (segs[:, :1] + t[None] * (segs[:, 1:] - segs[:, :1])).clamp(0, L - 1)
        lo = pos.floor().long(); hi = (lo + 1).clamp(max=L - 1); w = (pos - lo.float())[..., None]
        r0 = raw0.t()
        return r0[lo] * (1 - w) + r0[hi] * w
