"""Event segmentation + captioning architecture, with no classification layer.

Outputs:
  - event head     (B, T, 1) : is this time step inside an event
  - boundary head  (B, T, 2) : distances to the two boundaries of that event,
                               plus an IoU-quality score used for ranking
  - embedding head (B, T, D) : content vector in the same space as the captions
  - ground head    (Q, T)    : given a sentence, is this time step inside the event the sentence
                               describes (built when loss_weight_ground > 0). Its segments come
                               from the boundary head too, so a query only changes which steps win.

The three production inputs map onto these heads (uniav_api): a video alone -> events from the
event head, each captioned; a video and a sentence -> segments from the ground head; a sentence
alone -> the ground head over every stored video.

Inference: segments come from the event and boundary heads. Each segment's vector is
the mean embedding over its whole span (plus video context), compared by cosine with
the caption pool to pick a caption.

The embedding loss runs over the WHOLE train caption pool, not just the captions in
the batch, because inference must pick from that pool. Targets are soft: part of
the mass goes to the true caption and the rest is spread by caption-caption
similarity (in the caption space), so paraphrases of the same step are not punished
like wrong captions.

Optional OmniRetriever-7B teacher (omni_dim > 0), following the fusion-as-teacher
idea of the OmniRetriever paper: the segment vector is projected into its
3584-d space, then
  - pulled toward the frozen audio+video vector the 7B model gives the GT clip
  - contrasted with the caption pool in that space, with the same soft targets
At inference the scores of both spaces are added.

Options (train_cfg), all off by default:
  omni_target: 'tva'      the teacher target of a clip is its audio+video vector fused with the
                          vector of its caption (T+V+A), instead of audio+video alone
  loss_weight_modal: w    fusion-as-teacher inside the model (OmniRetriever's L_D): a segment
                          vector from the video stream alone and one from the audio stream alone
                          are each pulled toward the fused segment vector (stop-gradient) and
                          toward the right caption of the pool
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D, Scale, LayerNorm, Linear
from .multimodal_backbones import ConvTransformerBackbone
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss


def _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln):
    head, norm = nn.ModuleList(), nn.ModuleList()
    for i in range(n_layers - 1):
        head.append(MaskedConv1D(in_dim if i == 0 else feat_dim, feat_dim, ks,
                                 stride=1, padding=ks // 2, bias=(not with_ln)))
        norm.append(LayerNorm(feat_dim) if with_ln else nn.Identity())
    return head, norm


class EventHead(nn.Module):
    """A single channel: is there an event at this time step. Class agnostic."""

    def __init__(self, in_dim, feat_dim, n_layers=3, ks=3, with_ln=True, prior_prob=0.01):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)
        torch.nn.init.constant_(self.out.conv.bias, -np.log((1 - prior_prob) / prior_prob))

    def forward(self, fpn_feats, fpn_masks):
        res = tuple()
        for x, m in zip(fpn_feats, fpn_masks):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            o, _ = self.out(x, m)
            res += (o.permute(0, 2, 1),)          # (B, T, 1)
        return res


class EmbedHead(nn.Module):
    """One L2-normalised vector per time step, in the space of the projected captions."""

    def __init__(self, in_dim, feat_dim, n_layers=3, ks=3, with_ln=True, clip_dim=1536):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.vis_proj = MaskedConv1D(feat_dim, feat_dim, ks, stride=1, padding=ks // 2)
        self.clip_proj = Linear(clip_dim, feat_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, fpn_feats, fpn_masks):
        """Return the normalised vectors of every time step, plus the raw features of
        level 0 (finest resolution) for span averaging."""
        res, raw0 = tuple(), None
        for x, m in zip(fpn_feats, fpn_masks):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            x, _ = self.vis_proj(x, m)
            if raw0 is None:
                raw0 = x                                       # (B, D, T0)
            res += (F.normalize(x, dim=1).permute(0, 2, 1),)   # (B, T, D)
        return res, raw0

    def embed_captions(self, cap):
        return F.normalize(self.clip_proj(cap), dim=-1)


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

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim + 2, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, raw, length, segs):
        span = span_mean(raw, length, segs)                          # (N, D)
        glob = raw[:, :length].mean(-1).expand_as(span)              # (N, D)
        pos = (segs / float(length)).clamp(0, 1)                     # (N, 2)
        return F.normalize(span + self.mlp(torch.cat((span, glob, pos), -1)), dim=-1)


class BoundaryHead(nn.Module):
    def __init__(self, in_dim, feat_dim, fpn_levels, n_layers=3, ks=3, with_ln=True):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.scale = nn.ModuleList([Scale() for _ in range(fpn_levels)])
        self.out = MaskedConv1D(feat_dim, 2, ks, stride=1, padding=ks // 2)
        # predicts the IoU between the segment from this step and the GT, used for ranking
        self.iou_out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)

    def forward(self, fpn_feats, fpn_masks):
        res, qual = tuple(), tuple()
        for l, (x, m) in enumerate(zip(fpn_feats, fpn_masks)):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            o, _ = self.out(x, m)
            q, _ = self.iou_out(x, m)
            res += (F.relu(self.scale[l](o)).permute(0, 2, 1),)  # (B, T, 2)
            qual += (q.permute(0, 2, 1),)                         # (B, T, 1)
        return res, qual


class GroundHead(nn.Module):
    """Query-conditioned event head: is this time step inside the event the sentence describes.

    The query is a caption-space vector (InternVideo2 text, centred like the caption pool). It
    modulates the features after the first conv layer (FiLM: a per-channel scale and shift, both
    starting at zero), and the cosine between each step's embedding and the projected query is
    added to the logit, so the head starts from what the embedding head already knows.
    The first layer does not depend on the query, so it runs once per video, not once per query.
    """

    def __init__(self, in_dim, feat_dim, text_dim, n_layers=3, ks=3, with_ln=True, prior_prob=0.01):
        super().__init__()
        assert n_layers >= 2
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.film = nn.Linear(text_dim, 2 * feat_dim)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)
        torch.nn.init.constant_(self.out.conv.bias, -np.log((1 - prior_prob) / prior_prob))
        self.cos_scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, fpn_feats, fpn_masks, vid, query, query_e, em):
        """vid (Q,) video of each query in the batch, query (Q, text_dim) caption-space vectors,
        query_e (Q, D) the same projected into the event space, em (B, P, D) step embeddings.
        Returns logits (Q, P) over the steps of every pyramid level."""
        gamma, beta = self.film(query).chunk(2, dim=-1)                      # (Q, C)
        res = []
        for x, m in zip(fpn_feats, fpn_masks):
            x, _ = self.head[0](x, m)
            x = self.act(self.norm[0](x))[vid]                               # (Q, C, T)
            m = m[vid]
            x = x * (1 + gamma[:, :, None]) + beta[:, :, None]
            for i in range(1, len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            o, _ = self.out(x, m)
            res.append(o.squeeze(1))                                         # (Q, T)
        cos = (em[vid] * query_e[:, None]).sum(-1)                           # (Q, P)
        return torch.cat(res, dim=1) + self.cos_scale * cos


class EventCaptionTransformer(nn.Module):
    """Segment events, then caption them. No classification layer."""

    def __init__(self, backbone_type, backbone_arch, scale_factor, input_dim_V,
                 input_dim_A, n_head, embd_kernel_size, embd_dim, embd_with_ln,
                 head_dim, regression_range, head_num_layers, head_kernel_size,
                 head_with_ln, use_abs_pe, train_cfg, test_cfg, max_seq_len,
                 nmax=16, clip_dim=1536, omni_dim=0):
        super().__init__()
        self.fpn_strides = [scale_factor ** i for i in range(backbone_arch[-1] + 1)]
        assert len(self.fpn_strides) == len(regression_range)
        self.max_seq_len = max_seq_len
        self.max_div_factor = max(self.fpn_strides)
        self.nmax = nmax
        self.w_reg = train_cfg['loss_weight_reg']
        self.w_emb = train_cfg['loss_weight_emb']
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9
        self.test_score_thresh = test_cfg['score_thresh']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_max_seg = test_cfg['max_seg_num']
        self.nms_cfg = {'soft': test_cfg.get('soft_nms', True),
                        'iou': test_cfg.get('nms_iou', 0.7),
                        'sigma': test_cfg.get('nms_sigma', 0.5)}
        # embedding: soft targets over the whole pool, and span-averaged vectors
        self.w_span = train_cfg.get('loss_weight_span', 1.0)
        self.soft_alpha = train_cfg.get('emb_soft_alpha', 0.5)
        self.soft_tau = train_cfg.get('emb_soft_tau', 0.02)
        self.span_jitter = train_cfg.get('span_jitter', 0.2)
        self.max_pts = train_cfg.get('emb_max_points', 1024)
        self.w_iou = train_cfg.get('loss_weight_iou', 1.0)
        self.iou_power = test_cfg.get('iou_power', 0.5)
        self.w_omni_txt = train_cfg.get('loss_weight_omni_txt', 0.2)
        self.w_omni_av = train_cfg.get('loss_weight_omni_av', 0.2)
        self.omni_target = train_cfg.get('omni_target', 'av')
        self.w_modal = train_cfg.get('loss_weight_modal', 0.0)
        # grounding: every caption of a video is a query for its own span; captions of other videos
        # in the batch (not close to any caption of this video) are queries with no span
        self.w_ground = train_cfg.get('loss_weight_ground', 0.0)
        self.ground_neg = train_cfg.get('ground_neg_per_video', 2)
        self.ground_neg_max_cos = train_cfg.get('ground_neg_max_cos', 0.7)
        self.pool_raw = None
        self.omni_dim = omni_dim

        assert backbone_type == 'convTransformer', backbone_type
        self.backbone = ConvTransformerBackbone(
            n_in_V=input_dim_V, n_in_A=input_dim_A, n_embd=embd_dim, n_head=n_head,
            n_embd_ks=embd_kernel_size, max_len=max_seq_len, arch=backbone_arch,
            scale_factor=scale_factor, with_ln=embd_with_ln, attn_pdrop=0.0,
            proj_pdrop=train_cfg['dropout'], path_pdrop=train_cfg['droppath'], use_abs_pe=use_abs_pe)
        D = embd_dim * 2
        self.event_head = EventHead(D, head_dim, head_num_layers, head_kernel_size,
                                    head_with_ln, train_cfg['cls_prior_prob'])
        self.bound_head = BoundaryHead(D, head_dim, len(self.fpn_strides),
                                       head_num_layers, head_kernel_size, head_with_ln)
        self.embed_head = EmbedHead(D, head_dim, head_num_layers, head_kernel_size,
                                    head_with_ln, clip_dim)
        self.seg_ctx = SegmentContext(head_dim)
        if self.w_ground > 0:
            assert clip_dim == 512, 'the ground head takes InternVideo2 text queries (caption_space iv2)'
            self.ground_head = GroundHead(D, head_dim, clip_dim, head_num_layers, head_kernel_size,
                                          head_with_ln, train_cfg['cls_prior_prob'])
        if self.w_modal > 0:   # single-stream students, training only
            self.modal_proj = nn.ModuleDict({m: nn.Linear(embd_dim, head_dim) for m in ('v', 'a')})
        if omni_dim > 0:
            self.omni_proj = nn.Sequential(nn.Linear(head_dim, head_dim * 2), nn.GELU(),
                                           nn.Linear(head_dim * 2, omni_dim))
            self.omni_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def set_caption_pool(self, pool_raw):
        """Train caption pool (N, clip_dim) in the caption space, the negative set during training."""
        self.pool_raw = pool_raw
        self.pool_n = F.normalize(pool_raw, dim=-1)

    def set_omni_pool(self, text_pool, text_ok, av_pool, av_text_idx=None):
        """Teacher pools: caption vectors (N, d) aligned with the caption pool, and the
        audio+video vector of every GT clip (K, d); av_text_idx (K,) is each clip's caption."""
        self.omni_text = F.normalize(text_pool.float(), dim=-1)
        self.omni_text_ok = text_ok
        self.omni_av = F.normalize(av_pool.float(), dim=-1)
        if self.omni_target == 'tva':
            assert av_text_idx is not None
            self.omni_av = F.normalize(self.omni_av + self.omni_text[av_text_idx], dim=-1)

    def omni_embed(self, q):
        return F.normalize(self.omni_proj(q), dim=-1)

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        fv = [x['feats']['visual'] for x in video_list]
        fa = [x['feats']['audio'] for x in video_list]
        lens = torch.as_tensor([f.shape[-1] for f in fv])
        max_len = self.max_seq_len if self.training or lens.max() <= self.max_seq_len else \
            int((lens.max() + self.max_div_factor - 1) // self.max_div_factor * self.max_div_factor)
        def pad(xs):
            out = xs[0].new_full([len(xs), xs[0].shape[0], max_len], padding_val)
            for s, d in zip(xs, out):
                d[..., :s.shape[-1]].copy_(s)
            return out.to(self.device)
        masks = (torch.arange(max_len)[None, :] < lens[:, None]).unsqueeze(1).to(self.device)
        return pad(fv), pad(fa), masks

    def forward(self, video_list):
        V, A, masks = self.preprocessing(video_list)
        fV, fA, msk = self.backbone(V, A, masks)
        self._streams0 = (fV[0], fA[0])   # level-0 streams, for the single-stream students
        feats = [torch.cat((v, a), 1) for v, a in zip(fV, fA)]
        ev = torch.cat(self.event_head(feats, msk), dim=1).squeeze(-1)   # (B, P)
        bd, qu = self.bound_head(feats, msk)
        bd = torch.cat(bd, dim=1)                                        # (B, P, 2)
        qu = torch.cat(qu, dim=1).squeeze(-1)                            # (B, P)
        em, raw0 = self.embed_head(feats, msk)
        em = torch.cat(em, dim=1)                                        # (B, P, D)
        len0 = msk[0].squeeze(1).sum(-1)                                 # (B,)
        fpn_masks = torch.cat([m.squeeze(1) for m in msk], dim=1)        # (B, P)
        self._fpn = (feats, msk, em)                                     # for the ground head
        if self.training:
            return self.losses(video_list, fpn_masks, ev, bd, qu, em, raw0, len0)
        return self.inference(video_list, fpn_masks, ev, bd, qu, em, raw0, len0)

    def pool_loss(self, q, gt_idx, pool_f, log_scale=None, col_ok=None):
        """Cross-entropy over the whole pool with soft targets.

        q (M, D) normalised queries, gt_idx (M,) index of the true caption in the pool.
        col_ok (N,) marks which pool captions are usable.
        """
        log_scale = self.embed_head.logit_scale if log_scale is None else log_scale
        logit = log_scale.exp().clamp(max=100) * (q @ pool_f.t())
        with torch.no_grad():
            sim = self.pool_n[gt_idx] @ self.pool_n.t() / self.soft_tau
            if col_ok is not None:
                sim = sim.masked_fill(~col_ok[None], float('-inf'))
            tgt = self.soft_alpha * F.softmax(sim, dim=-1)
            tgt[torch.arange(len(gt_idx), device=q.device), gt_idx] += 1 - self.soft_alpha
        if col_ok is not None:
            logit = logit.masked_fill(~col_ok[None], -1e4)
        return -(tgt * F.log_softmax(logit, dim=-1)).sum(-1).mean()

    def losses(self, video_list, valid, ev, bd, qu, em, raw0, len0):
        dev = ev.device
        gt_cls = torch.stack([x['gt_cls_labels'].to(dev) for x in video_list])   # (B,P,NMAX)
        gt_off = torch.stack([x['gt_offsets'].to(dev) for x in video_list])      # (B,P,2)

        is_event = gt_cls.sum(-1) > 0
        pos = is_event & valid
        n_pos = int(pos.sum().item())
        self.loss_normalizer = (self.loss_normalizer_momentum * self.loss_normalizer
                                + (1 - self.loss_normalizer_momentum) * max(n_pos, 1))

        # 1. event or not: binary, no classes
        ev_loss = sigmoid_focal_loss(ev[valid], is_event[valid].float(),
                                     reduction='sum') / self.loss_normalizer
        # 2. segment boundaries
        if n_pos == 0:
            rg_loss = 0 * bd.sum()
        else:
            rg_loss = ctr_diou_loss_1d(bd[pos], gt_off[pos], reduction='sum',
                                       class_aware=False) / self.loss_normalizer
        # 2b. boundary quality: predict the true IoU of this prediction, used for ranking
        if n_pos == 0:
            iou_loss = 0 * qu.sum()
        else:
            p, g = bd[pos].detach(), gt_off[pos]
            inter = torch.min(p[:, 0], g[:, 0]) + torch.min(p[:, 1], g[:, 1])
            union = p.sum(-1) + g.sum(-1) - inter
            tgt = (inter / union.clamp(min=1e-6)).clamp(0, 1)
            iou_loss = F.binary_cross_entropy_with_logits(qu[pos], tgt)
        # 3. per-step embedding: contrast against the whole caption pool
        assert self.pool_raw is not None, 'call set_caption_pool before training'
        cap_pool = torch.stack([x['cap_pool_idx'] for x in video_list]).to(dev)  # (B,NMAX)
        pool_f = self.embed_head.embed_captions(self.pool_raw)                  # (N, D)
        if n_pos == 0:
            em_loss = 0 * em.sum()
        else:
            bi, pi = pos.nonzero(as_tuple=True)
            if bi.numel() > self.max_pts:
                sel = torch.randperm(bi.numel(), device=dev)[:self.max_pts]
                bi, pi = bi[sel], pi[sel]
            gt = cap_pool[bi, gt_cls[bi, pi].argmax(-1)]
            em_loss = self.pool_loss(em[bi, pi], gt, pool_f)
        # 4. span embedding: mean over the GT span, plus a jittered copy so the model
        #    copes with predicted boundaries that are slightly off at inference
        qs, gs, avs, zs = [], [], [], {'v': [], 'a': []}
        for b, x in enumerate(video_list):
            seg = x['segments'].to(dev).float()
            if seg.numel() == 0:
                continue
            L = int(len0[b])
            w = (seg[:, 1] - seg[:, 0]).clamp(min=1.0)
            j = self.span_jitter * w[:, None] * (2 * torch.rand_like(seg) - 1)
            for sg in (seg, seg + j):
                qs.append(self.seg_ctx(raw0[b], L, sg))
                if self.w_modal > 0:
                    for m, f in zip(('v', 'a'), self._streams0):
                        zs[m].append(F.normalize(self.modal_proj[m](span_mean(f[b], L, sg)), dim=-1))
                gs.append(cap_pool[b, x['labels'].to(dev)])
                if self.omni_dim > 0:
                    avs.append(x['cap_av_idx'].to(dev)[x['labels'].to(dev)])
        out = {'ev_loss': ev_loss, 'reg_loss': rg_loss, 'iou_loss': iou_loss,
               'emb_loss': em_loss}
        if qs:
            qs, gs = torch.cat(qs), torch.cat(gs)
            span_loss = self.pool_loss(qs, gs, pool_f)
        else:
            span_loss = 0 * raw0.sum()
        out['span_loss'] = span_loss
        total = (ev_loss + self.w_reg * rg_loss + self.w_iou * iou_loss
                 + self.w_emb * em_loss + self.w_span * span_loss)
        # fusion-as-teacher inside the model: each stream alone toward the fused vector and the caption
        if self.w_modal > 0 and len(gs):
            teacher = qs.detach()
            scale = self.embed_head.logit_scale.exp().clamp(max=100)
            ml = 0
            for m in ('v', 'a'):
                z = torch.cat(zs[m])
                ml = ml + F.cross_entropy(scale * z @ teacher.t(), torch.arange(len(z), device=dev))
                ml = ml + self.pool_loss(z, gs, pool_f)
            out['modal_loss'] = ml / 2
            total = total + self.w_modal * out['modal_loss']
        # 5. OmniRetriever teacher: captions in its space, and the GT clip as target
        if self.omni_dim > 0:
            zero = 0 * raw0.sum()
            ot, oa = zero, zero
            if len(gs):
                so = self.omni_embed(qs)
                rk = self.omni_text_ok[gs]          # skip spans whose caption has no teacher vector
                if rk.any():
                    ot = self.pool_loss(so[rk], gs[rk], self.omni_text, self.omni_logit_scale,
                                        self.omni_text_ok)
                ai = torch.cat(avs); ok = ai >= 0
                if ok.any():
                    la = self.omni_logit_scale.exp().clamp(max=100) * (so[ok] @ self.omni_av.t())
                    oa = F.cross_entropy(la, ai[ok])
            out['omni_txt_loss'], out['omni_av_loss'] = ot, oa
            total = total + self.w_omni_txt * ot + self.w_omni_av * oa
        # 6. grounding: each caption finds its own span, other videos' captions find none
        if self.w_ground > 0:
            out['ground_loss'] = self.ground_loss(video_list, gt_cls, cap_pool, valid)
            total = total + self.w_ground * out['ground_loss']
        out['final_loss'] = total
        return out

    def ground(self, vid, query):
        """Ground-head logits (Q, P) on the last forward pass; vid (Q,) batch index of each query,
        query (Q, 512) InternVideo2 caption vectors."""
        feats, msk, em = self._fpn
        q = query.float()
        return self.ground_head(feats, msk, vid, q, self.embed_head.embed_captions(q), em)

    def ground_loss(self, video_list, gt_cls, cap_pool, valid):
        dev = gt_cls.device
        vid, qi, tgt = [], [], []
        B = len(video_list)
        own = [cap_pool[b][cap_pool[b] >= 0] for b in range(B)]          # pool ids of each video's captions
        for b, x in enumerate(video_list):
            for j in x['labels'].unique().tolist():                       # captions still in the (cropped) clip
                vid.append(b); qi.append(int(cap_pool[b, j])); tgt.append(gt_cls[b, :, j])
            if B < 2 or self.ground_neg == 0:
                continue
            others = torch.cat([own[c] for c in range(B) if c != b])
            sim = (self.pool_n[others] @ self.pool_n[own[b]].t()).max(-1).values
            cand = others[sim < self.ground_neg_max_cos]
            for k in torch.randperm(len(cand), device=dev)[:self.ground_neg].tolist():
                vid.append(b); qi.append(int(cand[k])); tgt.append(torch.zeros_like(gt_cls[b, :, 0]))
        vid = torch.as_tensor(vid, device=dev)
        logit = self.ground(vid, self.pool_raw[torch.as_tensor(qi, device=dev)])
        tgt = torch.stack(tgt).clamp(max=1)
        ok = valid[vid]
        return sigmoid_focal_loss(logit[ok], tgt[ok], reduction='sum') / tgt[ok].sum().clamp(min=1)

    @torch.no_grad()
    def inference(self, video_list, valid, ev, bd, qu, em, raw0, len0):
        """Segment with event score times boundary-quality score, and return the embeddings."""
        from ..utils import batched_nms
        a = self.iou_power
        prob = ev.sigmoid().pow(1 - a) * qu.sigmoid().pow(a) * valid.float()
        out = []
        for b, v in enumerate(video_list):
            pts = torch.cat(v['points'], dim=0).to(ev.device)
            keep = (prob[b] > self.test_score_thresh).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                keep = prob[b].topk(1).indices
            sc = prob[b][keep]
            k = min(self.test_max_seg * 5, keep.numel())
            sc, order = sc.topk(k)
            keep = keep[order]
            p, o = pts[keep], bd[b][keep]
            left = p[:, 0] - o[:, 0] * p[:, 3]
            right = p[:, 0] + o[:, 1] * p[:, 3]
            segs = torch.stack((left, right), -1).cpu()
            scores = sc.cpu()
            emb = em[b][keep].cpu()
            ok = (segs[:, 1] - segs[:, 0]) > self.test_duration_thresh
            segs, scores, emb = segs[ok], scores[ok], emb[ok]
            # class-agnostic NMS: a single dummy label
            if segs.shape[0] > 0:
                idx = torch.arange(segs.shape[0])
                s2, sc2, idx2 = batched_nms(segs, scores, idx, self.nms_cfg['iou'], 0.001,
                                            self.test_max_seg,
                                            use_soft_nms=self.nms_cfg['soft'],
                                            multiclass=False, sigma=self.nms_cfg['sigma'],
                                            voting_thresh=0.0)
                emb = emb[idx2.long()]
                segs, scores = s2, sc2
            # segment vector = mean over the whole span, computed before converting to seconds
            if segs.shape[0] > 0:
                span = self.seg_ctx(raw0[b], int(len0[b]), segs.to(raw0.device))
            else:
                span = emb.to(raw0.device)
            omni = self.omni_embed(span).cpu() if self.omni_dim > 0 else None
            span = span.cpu()
            st, nf, fps = v['feat_stride'], v['feat_num_frames'], v['fps']
            # vectors of the GT spans themselves, to score captioning apart from segmentation
            gseg = v['segments'].to(raw0.device).float()
            gemb = self.seg_ctx(raw0[b], int(len0[b]), gseg)
            gomni = self.omni_embed(gemb).cpu() if self.omni_dim > 0 else None
            segs = (segs * st + 0.5 * nf) / fps
            segs = segs.clamp(min=0.0, max=float(v['duration']))
            gr = None
            if self.w_ground > 0:   # each GT caption as a query: its best segment, in seconds
                n = len(v['segments_sec'])
                gl = self.ground(torch.full((n,), b, device=ev.device), v['cap_emb'][:n].to(ev.device))
                gp = gl.sigmoid().pow(1 - a) * qu[b][None].sigmoid().pow(a) * valid[b][None].float()
                best = gp.argmax(-1)
                gpt = torch.cat(v['points'], dim=0).to(ev.device)[best]
                go = bd[b][best]
                gr = torch.stack((gpt[:, 0] - go[:, 0] * gpt[:, 3], gpt[:, 0] + go[:, 1] * gpt[:, 3]), -1).cpu()
                gr = ((gr * st + 0.5 * nf) / fps).clamp(min=0.0, max=float(v['duration']))
            out.append({'video_id': v['video_id'], 'segments': segs, 'ground_segments': gr,
                        'scores': scores, 'embeds': span, 'embeds_pt': emb,
                        'embeds_omni': omni, 'embeds_gt': gemb.cpu(), 'embeds_gt_omni': gomni,
                        'gt_segments': v['segments_sec'], 'gt_text': v['cap_text']})
        return out
