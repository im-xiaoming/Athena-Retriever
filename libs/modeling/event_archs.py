"""Kiến trúc tách sự kiện và sinh mô tả, không có lớp phân loại nào.

Ba đầu ra:
  - đầu sự kiện  (B, T, 1)   : mốc này có nằm trong một sự kiện không
  - đầu biên     (B, T, 2)   : khoảng cách tới hai biên của sự kiện đó
  - đầu nhúng    (B, T, D)   : vector mô tả nội dung, cùng không gian với caption

Lúc suy luận: tách đoạn bằng điểm sự kiện và biên, rồi với mỗi đoạn lấy trung
bình vector nhúng trên cả khoảng thời gian của đoạn, đem so cosine với kho caption
để ra câu mô tả.

Hàm mất mát của đầu nhúng so với TOÀN BỘ kho caption tập train chứ không chỉ các
caption trong lô, vì lúc suy luận model phải chọn đúng trong kho đó. Nhãn là phân
phối mềm: một phần dồn vào caption thật, phần còn lại trải theo độ giống nhau giữa
các caption (trong không gian ONE-PEACE), nên các câu diễn đạt khác của cùng một
thao tác không bị phạt như câu sai hẳn.

Tuỳ chọn thầy OmniRetriever-7B (omni_dim > 0), theo ý fusion-as-teacher của bài
OmniRetriever: vector đoạn được chiếu sang không gian 3584 chiều của nó rồi
  - kéo về vector audio+video mà model 7B sinh cho đúng clip GT (thầy, đã đóng băng)
  - so với kho caption trong không gian đó, cùng nhãn mềm như trên
Lúc suy luận cộng điểm của hai không gian.
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
        """Trả về vector đã chuẩn hoá của mọi mốc, kèm đặc trưng thô của tầng 0
        (độ phân giải mịn nhất) để lấy trung bình theo đoạn."""
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
    """Trung bình đặc trưng raw (D, T) trên các khoảng segs (N, 2) đơn vị lưới.

    Dùng tổng tích luỹ nên không phải lặp qua từng đoạn. Đoạn nào ngắn hơn một
    mốc vẫn lấy ít nhất một mốc.
    """
    cs = F.pad(raw.cumsum(-1), (1, 0))                         # (D, T+1)
    lo = segs[:, 0].floor().clamp(0, length - 1).long()
    hi = segs[:, 1].ceil().long() + 1
    hi = torch.maximum(hi.clamp(max=length), lo + 1)
    return (cs[:, hi] - cs[:, lo]).t() / (hi - lo).unsqueeze(1).float()   # (N, D)


class SegmentContext(nn.Module):
    """Vector của một đoạn, có thêm bối cảnh của cả video.

    Đầu vào là trung bình đặc trưng trên đoạn, trung bình trên cả video (đang nấu
    món gì) và vị trí tương đối của đoạn (bước đầu hay bước cuối công thức). Lớp
    cuối khởi tạo bằng 0 nên lúc đầu nó chính là trung bình theo đoạn.
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
        # du doan IoU giua doan sinh ra tu moc nay va doan that, dung de xep hang
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


@register_multimodal_meta_arch("EventCaptionTransformer")
class EventCaptionTransformer(nn.Module):
    """Tách sự kiện rồi sinh mô tả. Không có lớp phân loại nào."""

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
        # nhung: nhan mem tren toan kho, va trung binh theo doan
        self.w_span = train_cfg.get('loss_weight_span', 1.0)
        self.soft_alpha = train_cfg.get('emb_soft_alpha', 0.5)
        self.soft_tau = train_cfg.get('emb_soft_tau', 0.02)
        self.span_jitter = train_cfg.get('span_jitter', 0.2)
        self.max_pts = train_cfg.get('emb_max_points', 1024)
        self.w_iou = train_cfg.get('loss_weight_iou', 1.0)
        self.iou_power = test_cfg.get('iou_power', 0.5)
        self.w_omni_txt = train_cfg.get('loss_weight_omni_txt', 0.2)
        self.w_omni_av = train_cfg.get('loss_weight_omni_av', 0.2)
        self.pool_raw = None
        self.omni_dim = omni_dim

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
        self.seg_ctx = SegmentContext(head_dim)
        if omni_dim > 0:
            self.omni_proj = nn.Sequential(nn.Linear(head_dim, head_dim * 2), nn.GELU(),
                                           nn.Linear(head_dim * 2, omni_dim))
            self.omni_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def set_caption_pool(self, pool_raw):
        """Kho caption tập train (N, 1536), dùng làm tập mẫu âm khi huấn luyện."""
        self.pool_raw = pool_raw
        self.pool_n = F.normalize(pool_raw, dim=-1)

    def set_omni_pool(self, text_pool, text_ok, av_pool):
        """Kho của thầy: vector caption (N, d) thẳng hàng với kho caption, và
        vector audio+video của từng clip GT (K, d)."""
        self.omni_text = F.normalize(text_pool.float(), dim=-1)
        self.omni_text_ok = text_ok
        self.omni_av = F.normalize(av_pool.float(), dim=-1)

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
        fV, fA, msk = self.backbone(V, A, masks, 'TASK1', 'TAL')
        feats = [torch.cat((v, a), 1) for v, a in zip(fV, fA)]
        ev = torch.cat(self.event_head(feats, msk), dim=1).squeeze(-1)   # (B, P)
        bd, qu = self.bound_head(feats, msk)
        bd = torch.cat(bd, dim=1)                                        # (B, P, 2)
        qu = torch.cat(qu, dim=1).squeeze(-1)                            # (B, P)
        em, raw0 = self.embed_head(feats, msk)
        em = torch.cat(em, dim=1)                                        # (B, P, D)
        len0 = msk[0].squeeze(1).sum(-1)                                 # (B,)
        fpn_masks = torch.cat([m.squeeze(1) for m in msk], dim=1)        # (B, P)
        if self.training:
            return self.losses(video_list, fpn_masks, ev, bd, qu, em, raw0, len0)
        return self.inference(video_list, fpn_masks, ev, bd, qu, em, raw0, len0)

    def pool_loss(self, q, gt_idx, pool_f, log_scale=None, col_ok=None):
        """Cross-entropy trên toàn kho với nhãn mềm.

        q (M, D) đã chuẩn hoá, gt_idx (M,) chỉ số caption thật trong kho.
        col_ok (N,) đánh dấu caption nào trong kho dùng được.
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

        # 1. co su kien hay khong, nhi phan, khong co lop
        ev_loss = sigmoid_focal_loss(ev[valid], is_event[valid].float(),
                                     reduction='sum') / self.loss_normalizer
        # 2. bien doan
        if n_pos == 0:
            rg_loss = 0 * bd.sum()
        else:
            rg_loss = ctr_diou_loss_1d(bd[pos], gt_off[pos], reduction='sum',
                                       class_aware=False) / self.loss_normalizer
        # 2b. chat luong bien: hoc doan IoU that cua chinh du doan, dung de xep hang
        if n_pos == 0:
            iou_loss = 0 * qu.sum()
        else:
            p, g = bd[pos].detach(), gt_off[pos]
            inter = torch.min(p[:, 0], g[:, 0]) + torch.min(p[:, 1], g[:, 1])
            union = p.sum(-1) + g.sum(-1) - inter
            tgt = (inter / union.clamp(min=1e-6)).clamp(0, 1)
            iou_loss = F.binary_cross_entropy_with_logits(qu[pos], tgt)
        # 3. nhung tung moc: tuong phan voi toan kho caption
        assert self.pool_raw is not None, 'goi set_caption_pool truoc khi train'
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
        # 4. nhung theo doan: trung binh tren khoang GT, them mot ban bi xe dich
        #    de quen voi bien du doan khong khop hoan toan luc suy luan
        qs, gs, avs = [], [], []
        for b, x in enumerate(video_list):
            seg = x['segments'].to(dev).float()
            if seg.numel() == 0:
                continue
            L = int(len0[b])
            w = (seg[:, 1] - seg[:, 0]).clamp(min=1.0)
            j = self.span_jitter * w[:, None] * (2 * torch.rand_like(seg) - 1)
            for sg in (seg, seg + j):
                qs.append(self.seg_ctx(raw0[b], L, sg))
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
        # 5. thay OmniRetriever: caption trong khong gian cua no, va clip GT lam dich
        if self.omni_dim > 0:
            zero = 0 * raw0.sum()
            ot, oa = zero, zero
            if len(gs):
                so = self.omni_embed(qs)
                rk = self.omni_text_ok[gs]          # bo doan ma caption thieu vector thay
                if rk.any():
                    ot = self.pool_loss(so[rk], gs[rk], self.omni_text, self.omni_logit_scale,
                                        self.omni_text_ok)
                ai = torch.cat(avs); ok = ai >= 0
                if ok.any():
                    la = self.omni_logit_scale.exp().clamp(max=100) * (so[ok] @ self.omni_av.t())
                    oa = F.cross_entropy(la, ai[ok])
            out['omni_txt_loss'], out['omni_av_loss'] = ot, oa
            total = total + self.w_omni_txt * ot + self.w_omni_av * oa
        out['final_loss'] = total
        return out

    @torch.no_grad()
    def inference(self, video_list, valid, ev, bd, qu, em, raw0, len0):
        """Tách đoạn bằng điểm sự kiện nhân điểm chất lượng biên, trả kèm vector nhúng."""
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
            # NMS khong theo lop: dung mot nhan gia duy nhat
            if segs.shape[0] > 0:
                idx = torch.arange(segs.shape[0])
                s2, sc2, idx2 = batched_nms(segs, scores, idx, self.nms_cfg['iou'], 0.001,
                                            self.test_max_seg,
                                            use_soft_nms=self.nms_cfg['soft'],
                                            multiclass=False, sigma=self.nms_cfg['sigma'],
                                            voting_thresh=0.0)
                emb = emb[idx2.long()]
                segs, scores = s2, sc2
            # vector cua doan = trung binh tren ca khoang, tinh truoc khi doi ra giay
            if segs.shape[0] > 0:
                span = self.seg_ctx(raw0[b], int(len0[b]), segs.to(raw0.device))
            else:
                span = emb.to(raw0.device)
            omni = self.omni_embed(span).cpu() if self.omni_dim > 0 else None
            span = span.cpu()
            # vector cua chinh cac doan GT, de cham phan mo ta tach rieng khoi phan tach doan
            gseg = v['segments'].to(raw0.device).float()
            gemb = self.seg_ctx(raw0[b], int(len0[b]), gseg)
            gomni = self.omni_embed(gemb).cpu() if self.omni_dim > 0 else None
            st, nf, fps = v['feat_stride'], v['feat_num_frames'], v['fps']
            segs = (segs * st + 0.5 * nf) / fps
            segs = segs.clamp(min=0.0, max=float(v['duration']))
            out.append({'video_id': v['video_id'], 'segments': segs,
                        'scores': scores, 'embeds': span, 'embeds_pt': emb,
                        'embeds_omni': omni, 'embeds_gt': gemb.cpu(), 'embeds_gt_omni': gomni,
                        'gt_segments': v['segments_sec'], 'gt_text': v['cap_text']})
        return out
