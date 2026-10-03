"""Kiến trúc tách sự kiện và sinh mô tả, không có lớp phân loại nào.

Ba đầu ra:
  - đầu sự kiện  (B, T, 1)   : mốc này có nằm trong một sự kiện không
  - đầu biên     (B, T, 2)   : khoảng cách tới hai biên của sự kiện đó
  - đầu nhúng    (B, T, D)   : vector mô tả nội dung, cùng không gian với caption

Lúc suy luận: tách đoạn bằng điểm sự kiện và biên, rồi với mỗi đoạn lấy vector
nhúng đem so cosine với kho caption để ra câu mô tả gần nhất.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import register_multimodal_meta_arch, make_multimodal_backbone
from .blocks import MaskedConv1D, Scale, LayerNorm, Linear
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss


def _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln):
    head, norm = nn.ModuleList(), nn.ModuleList()
    for i in range(n_layers - 1):
        head.append(MaskedConv1D(in_dim if i == 0 else feat_dim, feat_dim, ks,
                                 stride=1, padding=ks // 2, bias=(not with_ln)))
        norm.append(LayerNorm(feat_dim) if with_ln else nn.Identity())
    return head, norm


class EventHead(nn.Module):
    """Một kênh duy nhất: mốc này có sự kiện hay không. Không biết lớp."""

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
    """Mỗi mốc một vector đã chuẩn hoá L2, cùng không gian với caption đã chiếu."""

    def __init__(self, in_dim, feat_dim, n_layers=3, ks=3, with_ln=True, clip_dim=1536):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.vis_proj = MaskedConv1D(feat_dim, feat_dim, ks, stride=1, padding=ks // 2)
        self.clip_proj = Linear(clip_dim, feat_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, fpn_feats, fpn_masks):
        res = tuple()
        for x, m in zip(fpn_feats, fpn_masks):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            x, _ = self.vis_proj(x, m)
            res += (F.normalize(x, dim=1).permute(0, 2, 1),)   # (B, T, D)
        return res

    def embed_captions(self, cap):
        return F.normalize(self.clip_proj(cap), dim=-1)


class BoundaryHead(nn.Module):
    def __init__(self, in_dim, feat_dim, fpn_levels, n_layers=3, ks=3, with_ln=True):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.scale = nn.ModuleList([Scale() for _ in range(fpn_levels)])
        self.out = MaskedConv1D(feat_dim, 2, ks, stride=1, padding=ks // 2)

    def forward(self, fpn_feats, fpn_masks):
        res = tuple()
        for l, (x, m) in enumerate(zip(fpn_feats, fpn_masks)):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            o, _ = self.out(x, m)
            res += (F.relu(self.scale[l](o)).permute(0, 2, 1),)  # (B, T, 2)
        return res


@register_multimodal_meta_arch("EventCaptionTransformer")
class EventCaptionTransformer(nn.Module):
    """Tách sự kiện rồi sinh mô tả. Không có lớp phân loại nào."""

    def __init__(self, backbone_type, backbone_arch, scale_factor, input_dim_V,
                 input_dim_A, n_head, embd_kernel_size, embd_dim, embd_with_ln,
                 head_dim, regression_range, head_num_layers, head_kernel_size,
                 head_with_ln, use_abs_pe, train_cfg, test_cfg, max_seq_len,
                 nmax=16, clip_dim=1536):
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

        self.backbone = make_multimodal_backbone('convTransformer', **{
            'n_in_V': input_dim_V, 'n_in_A': input_dim_A, 'n_embd': embd_dim,
            'n_head': n_head, 'n_embd_ks': embd_kernel_size,
            'max_len': {'TASK1': max_seq_len}, 'arch': backbone_arch,
            'scale_factor': scale_factor, 'with_ln': embd_with_ln,
            'attn_pdrop': 0.0, 'proj_pdrop': train_cfg['dropout'],
            'path_pdrop': train_cfg['droppath'], 'use_abs_pe': use_abs_pe})
        D = embd_dim * 2
        self.event_head = EventHead(D, head_dim, head_num_layers, head_kernel_size,
                                    head_with_ln, train_cfg['cls_prior_prob'])
        self.bound_head = BoundaryHead(D, head_dim, len(self.fpn_strides),
                                       head_num_layers, head_kernel_size, head_with_ln)
        self.embed_head = EmbedHead(D, head_dim, head_num_layers, head_kernel_size,
                                    head_with_ln, clip_dim)

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
        fV, fA, msk = self.backbone(V, A, masks, 'TASK1', 'TAL')
        feats = [torch.cat((v, a), 1) for v, a in zip(fV, fA)]
        ev = torch.cat(self.event_head(feats, msk), dim=1).squeeze(-1)   # (B, P)
        bd = torch.cat(self.bound_head(feats, msk), dim=1)               # (B, P, 2)
        em = torch.cat(self.embed_head(feats, msk), dim=1)               # (B, P, D)
        fpn_masks = torch.cat([m.squeeze(1) for m in msk], dim=1)        # (B, P)
        if self.training:
            return self.losses(video_list, fpn_masks, ev, bd, em)
        return self.inference(video_list, fpn_masks, ev, bd, em)

    def losses(self, video_list, valid, ev, bd, em):
        dev = ev.device
        gt_cls = torch.stack([x['gt_cls_labels'].to(dev) for x in video_list])   # (B,P,NMAX)
        gt_off = torch.stack([x['gt_offsets'].to(dev) for x in video_list])      # (B,P,2)
        cap = torch.stack([x['cap_emb'] for x in video_list]).to(dev)            # (B,NMAX,1536)
        cmask = torch.stack([x['cap_mask'] for x in video_list]).to(dev)         # (B,NMAX)

        is_event = gt_cls.sum(-1) > 0
        pos = is_event & valid
        n_pos = int(pos.sum().item())
        self.loss_normalizer = (self.loss_normalizer_momentum * self.loss_normalizer
                                + (1 - self.loss_normalizer_momentum) * max(n_pos, 1))

        # 1. co su kien hay khong, nhi phan, khong co lop
        ev_loss = sigmoid_focal_loss(ev[valid], is_event[valid].float(),
                                     reduction='sum') / self.loss_normalizer
        # 2. bien doan
        if n_pos == 0:
            rg_loss = 0 * bd.sum()
        else:
            rg_loss = ctr_diou_loss_1d(bd[pos], gt_off[pos], reduction='sum',
                                       class_aware=False) / self.loss_normalizer
        # 3. nhung: tuong phan voi MOI caption trong lo
        B = cap.shape[0]
        capf = self.embed_head.embed_captions(cap).reshape(B * self.nmax, -1)    # (B*NMAX, D)
        flat_ok = cmask.reshape(-1)
        if n_pos == 0 or int(flat_ok.sum()) == 0:
            em_loss = 0 * em.sum()
        else:
            bi, pi = pos.nonzero(as_tuple=True)
            q = em[bi, pi]                                                       # (M, D)
            tgt = bi * self.nmax + gt_cls[bi, pi].argmax(-1)                     # (M,)
            logit = self.embed_head.logit_scale.exp() * (q @ capf.t())           # (M, B*NMAX)
            logit = logit.masked_fill(~flat_ok[None, :], torch.finfo(logit.dtype).min)
            em_loss = F.cross_entropy(logit, tgt)
        total = ev_loss + self.w_reg * rg_loss + self.w_emb * em_loss
        return {'ev_loss': ev_loss, 'reg_loss': rg_loss, 'emb_loss': em_loss,
                'final_loss': total}

    @torch.no_grad()
    def inference(self, video_list, valid, ev, bd, em):
        """Tách đoạn bằng điểm sự kiện, trả kèm vector nhúng của từng đoạn."""
        from ..utils import batched_nms
        prob = ev.sigmoid() * valid.float()
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
            # NMS khong theo lop: dung mot nhan gia duy nhat
            if segs.shape[0] > 0:
                idx = torch.arange(segs.shape[0])
                s2, sc2, idx2 = batched_nms(segs, scores, idx, 0.7, 0.001,
                                            self.test_max_seg, use_soft_nms=True,
                                            multiclass=False, sigma=0.5, voting_thresh=0.0)
                emb = emb[idx2.long()]
                segs, scores = s2, sc2
            st, nf, fps = v['feat_stride'], v['feat_num_frames'], v['fps']
            segs = (segs * st + 0.5 * nf) / fps
            segs = segs.clamp(min=0.0, max=float(v['duration']))
            out.append({'video_id': v['video_id'], 'segments': segs,
                        'scores': scores, 'embeds': emb,
                        'gt_segments': v['segments_sec'], 'gt_text': v['cap_text']})
        return out
