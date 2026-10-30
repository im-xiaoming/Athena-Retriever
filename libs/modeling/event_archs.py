# in_dim=1024
# feat_dim=512
# clip_dim=512


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import register_multimodal_meta_arch, make_multimodal_backbone
from .blocks import MaskedConv1D, Scale, LayerNorm, Linear
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss
from ..utils import batched_nms


def _conv_stack(
    in_dim,     # channels of the input feature pyramid (2 * embd_dim: video stream + audio stream)
    feat_dim,   # channels inside the head
    n_layers,   # layers of the head, output conv included (so n_layers - 1 are built here)
    ks,         # conv kernel size
    with_ln,    # LayerNorm after every conv
): # not change T
    """Shared body of the heads: n_layers - 1 MaskedConv1D layers, each with its LayerNorm.
    Each head adds its own ReLU and output conv."""
    head, norm = nn.ModuleList(), nn.ModuleList()
    for i in range(n_layers - 1):
        head.append(MaskedConv1D(in_dim if i == 0 else feat_dim, feat_dim, ks,
                                 stride=1, padding=ks // 2, bias=(not with_ln)))
        norm.append(LayerNorm(feat_dim) if with_ln else nn.Identity())
    return head, norm


class EventHead(nn.Module):
    """
    Iterate for each level of T = [256, 128, 64, 32, 16, 8], and check if each level has event occured.

    Returns:
        Tuple[tensor[B, T, 1]]
    """

    def __init__(
        self,
        in_dim,           # channels of the input feature pyramid
        feat_dim,         # channels inside the head
        n_layers=3,       # conv layers, output conv included
        ks=3,             # kernel size
        with_ln=True,     # LayerNorm after every conv
        prior_prob=0.01,  # probability of "event" at init: output bias = -log((1 - p) / p)
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)
        torch.nn.init.constant_(self.out.conv.bias, -np.log((1 - prior_prob) / prior_prob))

    def forward(
        self,
        fpn_feats,  # 6 levels, each (B, C, T_l)
        fpn_masks,  # 6 levels, each (B, 1, T_l) bool, True = real step (not padding)
    ):
        res = tuple()
        for x, m in zip(fpn_feats, fpn_masks):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            o, _ = self.out(x, m)
            res += (o.permute(0, 2, 1),)          # (B, T_l, 1)
        return res # tuple[tensor(B, T_l, 1)]


class EmbedHead(nn.Module):
    """One L2-normalised vector per time step, in the same space as the projected captions.
    With omi_feats_dim > 0 a second projection puts every step in the OmniRetriever caption space too."""

    def __init__(
        self,
        in_dim,         # channels of the input feature pyramid
        feat_dim,       # channels inside the head, also the size D of every step vector
        n_layers=3,     # conv layers
        ks=3,           # kernel size
        with_ln=True,   # LayerNorm after every conv
        clip_dim=512,   # size of the caption vectors (InternVideo2 text)
        omi_feats_dim=3584
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.vis_proj = MaskedConv1D(feat_dim, feat_dim, ks, stride=1, padding=ks // 2)
        self.omni_vis_proj = None
        if omi_feats_dim > 0:
            self.omni_vis_proj = MaskedConv1D(feat_dim, omi_feats_dim, ks, stride=1, padding=ks // 2)
        self.clip_proj = Linear(clip_dim, feat_dim) # 512 -> 512 : captions
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        fpn_feats,  # 6 levels, each (B, C, T_l)
        fpn_masks,  # 6 levels, each (B, 1, T_l) bool
    ):
        """Returns the normalised vectors of every step (and of the omni projection, empty without it),
        and the raw features of level 0 (the finest) for the span averages."""
        res, raw0 = tuple(), None
        omni_res = tuple()
        for x, m in zip(fpn_feats, fpn_masks):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            x1, _ = self.vis_proj(x, m)
            if raw0 is None:
                raw0 = x1                                       # (B, D, T0)
            res += (F.normalize(x1, dim=1).permute(0, 2, 1),)   # (B, T_l, D)
            if self.omni_vis_proj is not None:
                x2, _ = self.omni_vis_proj(x, m)
                omni_res += (F.normalize(x2, dim=1).permute(0, 2, 1),)
        return res, omni_res, raw0

    def embed_captions(
        self,
        cap,  # (N, 512) vector caption
    ):
        return F.normalize(self.clip_proj(cap), dim=-1) # (N, 512)



# raw (512, 256) level 0, not normalized

def span_mean(
    raw,     # (D, T) level-0 features of one video, not normalised
    length,  # real steps (padding excluded) of the video
    segs,    # (N, 2) segments (start, end), in level-0 steps
):
    """Mean of the features over every segment.

    Uses a cumulative sum, so there is no loop over the segments. A segment shorter than one step
    still takes at least one step.
    """
    cs = F.pad(raw.cumsum(-1), (1, 0))                         # (D, T+1)
    lo = segs[:, 0].floor().clamp(min=0, max=length - 1).long()
    hi = segs[:, 1].ceil().long() + 1
    hi = torch.maximum(hi.clamp(max=length), lo + 1)
    return (cs[:, hi] - cs[:, lo]).t() / (hi - lo).unsqueeze(1).float()   # (N, D)


class SegmentContext(nn.Module):
    """Vector of one segment, with the context of the whole video.

    The input is the mean over the segment, the mean over the whole video (which task it is) and the
    relative position of the segment (first or last step of the task). The last layer starts at 0,
    so at first the output is exactly the segment mean.
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

    def forward(
        self,
        raw,     # (D, T) level-0 features of one video
        length,  # real steps of the video
        segs,    # (N, 2) segments, in steps
    ):
        span = span_mean(raw, length, segs)                          # (N, D)
        glob = raw[:, :length].mean(-1).expand_as(span)              # (N, D)
        pos = (segs / float(length)).clamp(0, 1)                     # (N, 2)
        return F.normalize(span + self.mlp(torch.cat((span, glob, pos), -1)), dim=-1)


class BoundaryHead(nn.Module):
    """At every step: distances to the start and end of the event holding it (in strides of the level,
    always >= 0), and an IoU-quality logit (how good the segment predicted from this step is, for ranking)."""

    def __init__(
        self,
        in_dim,          # channels of the input feature pyramid
        feat_dim,        # channels inside the head
        fpn_levels=6,    # pyramid levels: each level has its own scale for the distances
        n_layers=3,      # conv layers
        ks=3,            # kernel size
        with_ln=True,    # LayerNorm after every conv
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.scale = nn.ModuleList([Scale() for _ in range(fpn_levels)])
        self.out = MaskedConv1D(feat_dim, 2, ks, stride=1, padding=ks // 2)
        torch.nn.init.constant_(self.out.conv.bias, 1.0)
        # predicted IoU between the segment from this step and the GT segment, for ranking
        self.iou_out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)

    def forward(
        self,
        fpn_feats,  # 6 levels, each (B, C, T_l)
        fpn_masks,  # 6 levels, each (B, 1, T_l) bool
    ):
        res, qual = tuple(), tuple()
        for l, (x, m) in enumerate(zip(fpn_feats, fpn_masks)):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            q, _ = self.iou_out(x, m)
            qual += (q.permute(0, 2, 1),)                         # (B, T_l, 1)
            o, _ = self.out(x, m)
            res += (F.relu(self.scale[l](o)).permute(0, 2, 1),)  # (B, T_l, 2)
        return res, qual



@register_multimodal_meta_arch("EventCaptionTransformer")
class EventCaptionTransformer(nn.Module):
    """Finds the steps of a video and picks a caption for each. No classification layer."""

    def __init__(
        self,
        backbone_arch,       # (#conv, #transformer in the stem, #transformer in the pyramid branch), e.g. (2, 3, 5)
        scale_factor,        # length ratio between two pyramid levels
        input_dim_V,         # video feature channels (768)
        input_dim_A,         # audio feature channels (768)
        n_head,              # attention heads in the backbone
        embd_kernel_size,    # kernel size of the embedding convs in the backbone
        embd_dim,            # channels per backbone stream (512); the heads read both streams: 2 * embd_dim
        embd_with_ln,        # LayerNorm after the embedding convs
        head_dim,            # channels in every head, also the size D of the step / segment vectors
        omni_head_dim,       # size of the OmniRetriever caption vectors (3584), 0 = no omni head
        regression_range,    # each level handles the segments whose largest distance to a boundary falls in this range
        head_num_layers,     # conv layers of every head
        head_kernel_size,    # kernel size of the heads
        head_with_ln,        # LayerNorm in the heads
        use_abs_pe,          # add a sinusoid position embedding in the backbone
        train_cfg,           # loss weights, dropout, droppath, cls_prior_prob
        test_cfg,            # inference settings: thresholds, maximum segments, NMS
        max_seq_len,         # steps of every video (256)
        nmax=16,             # at most this many captions per video
        clip_dim=512,        # size of the caption vectors,
    ):
        super().__init__()
        self.fpn_strides = [scale_factor ** i for i in range(backbone_arch[-1] + 1)] # [1, 2, 4, 8, 16, 32]
        self.max_seq_len = max_seq_len
        self.nmax = nmax
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.backbone = make_multimodal_backbone(
            'convTransformer',
            **{
                'n_in_V': input_dim_V, 'n_in_A': input_dim_A, 'n_embd': embd_dim, 'n_head': n_head,
                'n_embd_ks': embd_kernel_size, 'max_len': max_seq_len, 'arch': backbone_arch,
                'scale_factor': scale_factor, 'with_ln': embd_with_ln, 'attn_pdrop': 0.0,
                'proj_pdrop': train_cfg['dropout'], 'path_pdrop': train_cfg['droppath'], 'use_abs_pe': use_abs_pe,
            }
        )
        D = embd_dim * 2 # 512 * 2
        self.event_head = EventHead(D, head_dim, head_num_layers, head_kernel_size,
                                    head_with_ln, train_cfg['cls_prior_prob'])
        self.bound_head = BoundaryHead(D, head_dim, len(self.fpn_strides),
                                       head_num_layers, head_kernel_size, head_with_ln)
        self.embed_head = EmbedHead(D, head_dim, head_num_layers, head_kernel_size,
                                    head_with_ln, clip_dim, omni_head_dim)
        self.seg_ctx = SegmentContext(head_dim)

    def set_caption_pool(
        self,
        pool_raw,  # (N, clip_dim) InternVideo2 vectors of the distinct training captions
        omni_pool_raw=None,  # (N, omni_head_dim) their OmniRetriever vectors, None = no omni head
    ):
        self.pool_raw = pool_raw # pool_emb
        self.omni_pool_raw = omni_pool_raw

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

    @torch.no_grad()
    def preprocessing(
        self,
        video_list,        # B dicts, one per video (HTStepDataset.__getitem__)
        padding_val=0.0,   # value of the padding steps
    ):
        """Stack the videos of the batch into (B, C=768, T) tensors and a (B, 1, T) mask.
        T is always max_seq_len: videos cropped during training are padded."""

        fv = [x['feats']['visual'] for x in video_list] # x['feats']['visual'] = (C, T')
        fa = [x['feats']['audio'] for x in video_list]
        lens = torch.as_tensor([f.shape[-1] for f in fv]) # T are different because of data agumnetation
        max_len = self.max_seq_len

        def pad(xs):
            """Padding function"""
            # new_full: create tensor same device and dtype
            #                        B           C           T
            out = xs[0].new_full([len(xs), xs[0].shape[0], max_len], padding_val)
            for s, d in zip(xs, out):
                d[..., :s.shape[-1]].copy_(s)
            return out.to(self.device)

        # B: bactch size
        masks = (torch.arange(max_len)[None, :] < lens[:, None]).unsqueeze(1).to(self.device) # (1, T) < (B, 1) = (B, T) -> (B, 1, T)
        return pad(fv), pad(fa), masks # (B, C, T)

    def forward(self, video_list):
        V, A, masks = self.preprocessing(video_list) # V = A = (B, C, T)

        # RETURN PYRAMID FEATURE
        #     tuple[(B, 1, T]
        fV, fA, msk = self.backbone(V, A, masks) # fV = fA = (B, C, T_l), l ∈ [0, 5]

        feats = [torch.cat((v, a), dim=1) for v, a in zip(fV, fA)]           # per level (B, 2 * embd_dim[512], T_l), len(feats) = 6

        # Event Head
        #                   tuple(tensor(B, T_l, 1))
        ev = torch.cat(self.event_head(feats, msk), dim=1).squeeze(-1)   # (B, P), P = point = 504

        # Bounding Head
        # tuple[tensor(B, T_l, 2))], tuple[tensor[(B, T_l, 1)]]
        bd, qu = self.bound_head(feats, msk)
        bd = torch.cat(bd, dim=1)                                        # (B, P, 2)
        qu = torch.cat(qu, dim=1).squeeze(-1)                            # (B, P)

        # Embedding Head
        em, omni_em, raw0 = self.embed_head(feats, msk) # tuple((B, T_l, D)), (B, D, T0)
        em = torch.cat(em, dim=1)                                        # (B, P, D)
        omni_em = torch.cat(omni_em, dim=1) if omni_em else None         # (B, P, 3584)

        #       (B, 1, T)
        len0 = msk[0].squeeze(1).sum(-1)                                 # (B,)
        #                         (B, T)
        fpn_masks = torch.cat([m.squeeze(1) for m in msk], dim=1)        # (B, P)
        if self.training:
            return self.losses(video_list, fpn_masks, ev, bd, qu, em, omni_em, raw0, len0)
        return self.inference(video_list, fpn_masks, ev, bd, qu, em, raw0, len0)

    def losses(
        self,
        video_list,  # B dicts, one per video
        valid,       # (B, P) bool, True at points of real steps (not padding)
        ev,          # (B, P) event head logits
        bd,          # (B, P, 2) predicted distances to the start / end, in strides
        qu,          # (B, P) IoU-quality logits
        em,          # (B, P, D) normalised embedding of every step
        omni_em,     # (B, P, D)
        raw0,        # (B, D, T) level-0 features of the embedding head, not normalised (for the span means)
        len0,        # (B,) real level-0 steps of every video
    ):
        dev = ev.device
        w = self.train_cfg
        gt_cls = torch.stack([x['gt_cls_labels'].to(dev) for x in video_list])   # (B, P, 16)
        gt_off = torch.stack([x['gt_offsets'].to(dev) for x in video_list])      # (B, P, 2)
        cap_pool = torch.stack([x['cap_pool_idx'] for x in video_list]).to(dev)  # (B, 16)

        is_event = gt_cls.sum(-1) > 0                                    # (B, P)
        pos = is_event & valid # (B, P)
        n_pos = max(int(pos.sum()), 1) #  >= 1

        #                               (V,)          (V,)
        ev_loss = sigmoid_focal_loss(ev[valid], is_event[valid].float(), reduction='sum') / n_pos

        #                           (N, 2)     (N, 2)
        rg_loss, iouk = ctr_diou_loss_1d(bd[pos], gt_off[pos], reduction='sum')
        rg_loss = rg_loss / n_pos

        # IoU-quality
        iou_loss = F.binary_cross_entropy_with_logits(qu[pos], iouk)

        # InfoNCE Loss
        pool_f = self.embed_head.embed_captions(self.pool_raw)          # (N, 512)
        scale = self.embed_head.logit_scale.exp().clamp(max=100)
        ri, ci = pos.nonzero(as_tuple=True)

        gt = cap_pool[ri, gt_cls[ri, ci].argmax(-1)] # (N',)
        #                                   (N', 512)    (512, N)
        emb_loss = F.cross_entropy(scale * em[ri, ci] @ pool_f.t(), gt)
        #                                       cosin matrix
        # Omni
        # the OmniRetriever caption vectors are not projected: the steps are projected into their space
        omni_emb_loss = torch.zeros((), device=dev)
        if omni_em is not None:
            pool_o = F.normalize(self.omni_pool_raw, dim=-1)                # (N, 3584)
            omni_emb_loss = F.cross_entropy(scale * omni_em[ri, ci] @ pool_o.t(), gt)


        # 5. span embedding: the vector of every GT segment must pick its own caption out of the whole pool
        qs, gs = [], []
        for i, x in enumerate(video_list):
            #                       (D, T)
            qs.append(self.seg_ctx(raw0[i], int(len0[i]), x['segments'].to(dev).float()))
            gs.append(cap_pool[i, x['labels'].to(dev)])
        span_loss = F.cross_entropy(scale * torch.cat(qs) @ pool_f.t(), torch.cat(gs))


        total = (ev_loss + w['loss_weight_reg'] * rg_loss + w['loss_weight_iou'] * iou_loss
                 + w['loss_weight_emb'] * emb_loss + w['loss_weight_omni_emb'] * omni_emb_loss
                 + w['loss_weight_span'] * span_loss)

        return {'ev_loss': ev_loss, 'reg_loss': rg_loss, 'iou_loss': iou_loss, 'emb_loss': emb_loss,
                'omni_emb_loss': omni_emb_loss, 'span_loss': span_loss, 'final_loss': total}

    @torch.no_grad()
    def inference(
        self,
        video_list,  # B dicts, one per video
        valid,       # (B, P) bool, True at points of real steps
        ev,          # (B, P) event head logits
        bd,          # (B, P, 2) predicted distances to the start / end, in strides
        qu,          # (B, P) IoU-quality logits
        em,          # (B, P, D) normalised embedding of every step
        raw0,        # (B, D, T) level-0 features of the embedding head (for the segment vectors)
        len0,        # (B,) real level-0 steps of every video
    ):
        """Score of a segment = event score times IoU-quality score. Returns one dict per video:
        segments (K, 2) in seconds, best first; embeds (K, D) the segment vectors;
        gt_segments, gt_text the GT steps (seconds) and their captions.
        """
        t = self.test_cfg
        a = t['iou_power']
        prob = ev.sigmoid().pow(1 - a) * qu.sigmoid().pow(a) * valid.float() # (B, P)
        out = []
        for b, v in enumerate(video_list):
            pts = torch.cat(v['points'], dim=0).to(ev.device)   # (P, 4): t, regression range low, high, stride
            #          (P,)
            keep = (prob[b] > t['score_thresh']).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                keep = prob[b].topk(1).indices
            sc, order = prob[b][keep].topk(min(t['max_seg_num'] * 5, keep.numel()))
            keep = keep[order]
            # segment of a point: from t - (distance to start) * stride to t + (distance to end) * stride
            p, o = pts[keep], bd[b][keep]
            segs = torch.stack((p[:, 0] - o[:, 0] * p[:, 3], p[:, 0] + o[:, 1] * p[:, 3]), -1).cpu()
            scores = sc.cpu()
            ok = (segs[:, 1] - segs[:, 0]) > t['duration_thresh']
            segs, scores = segs[ok], scores[ok]
            # class-agnostic soft-NMS: every segment shares one dummy label
            segs, scores, _ = batched_nms(segs, scores, torch.arange(len(segs)), 0.7, 0.001,
                                          t['max_seg_num'], use_soft_nms=True, sigma=t['nms_sigma'])
            # vector of the segment = mean over the whole segment, computed while it is still in steps
            span = self.seg_ctx(raw0[b], int(len0[b]), segs.to(raw0.device)).cpu()

            # steps to seconds (see __getitem__ of the dataset)
            st, nf, fps = v['feat_stride'], v['feat_num_frames'], v['fps']
            segs = ((segs * st + 0.5 * nf) / fps).clamp(min=0.0, max=float(v['duration']))
            out.append({'video_id': v['video_id'], 'segments': segs, 'scores': scores, 'embeds': span,
                        'gt_segments': v['segments_sec'], 'gt_text': v['cap_text']})
        return out
