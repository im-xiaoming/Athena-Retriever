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
    in_dim,     # số kênh của feature pyramid đầu vào (2 * embd_dim: luồng video + luồng audio)
    feat_dim,   # số kênh bên trong head
    n_layers,   # số lớp của head, tính cả conv đầu ra (nên ở đây chỉ tạo n_layers - 1 lớp)
    ks,         # kernel size của conv
    with_ln,    # có LayerNorm sau mỗi conv không
): # not change T
    """Phần thân chung của các head: n_layers - 1 lớp MaskedConv1D, mỗi lớp có LayerNorm đi kèm.
    ReLU và conv đầu ra do từng head tự thêm."""
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
        in_dim,           # số kênh của feature pyramid đầu vào
        feat_dim,         # số kênh bên trong head
        n_layers=3,       # số lớp conv, tính cả conv đầu ra
        ks=3,             # kernel size
        with_ln=True,     # LayerNorm sau mỗi conv
        prior_prob=0.01,  # xác suất "là event" lúc khởi tạo: bias đầu ra = -log((1 - p) / p)
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)
        torch.nn.init.constant_(self.out.conv.bias, -np.log((1 - prior_prob) / prior_prob))

    def forward(
        self,
        fpn_feats,  # list 6 tầng, mỗi tầng (B, C, T_l)
        fpn_masks,  # list 6 tầng, mỗi tầng (B, 1, T_l) kiểu bool, True = bước thật (không phải padding)
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
    """Mỗi bước thời gian ra một vector đã chuẩn hóa L2, nằm cùng không gian với caption đã chiếu."""

    def __init__(
        self,
        in_dim,         # số kênh của feature pyramid đầu vào
        feat_dim,       # số kênh bên trong head, cũng là kích thước D của vector mỗi bước
        n_layers=3,     # số lớp conv
        ks=3,           # kernel size
        with_ln=True,   # LayerNorm sau mỗi conv
        clip_dim=512,   # kích thước vector caption (text InternVideo2)
        omi_feats_dim=3584
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.vis_proj = MaskedConv1D(feat_dim, feat_dim, ks, stride=1, padding=ks // 2)
        self.omni_vis_proj = MaskedConv1D(feat_dim, omi_feats_dim, ks, stride=1, padding=ks // 2)
        self.clip_proj = Linear(clip_dim, feat_dim) # 512 -> 512 : captions
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        fpn_feats,  # list 6 tầng, mỗi tầng (B, C, T_l)
        fpn_masks,  # list 6 tầng, mỗi tầng (B, 1, T_l) kiểu bool
    ):
        """Trả về vector đã chuẩn hóa của mọi bước, và feature thô của tầng 0 (tầng mịn nhất)
        để lấy trung bình theo đoạn."""
        res, raw0 = tuple(), None
        omni_res = tuple()
        for x, m in zip(fpn_feats, fpn_masks):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            x1, _ = self.vis_proj(x, m)
            x2, _ = self.omni_vis_proj(x, m)
            if raw0 is None:
                raw0 = x1                                       # (B, D, T0)
            res += (F.normalize(x1, dim=1).permute(0, 2, 1),)   # (B, T_l, D)
            omni_res += (F.normalize(x2, dim=1).permute(0, 2, 1))
        return res, omni_res, raw0

    def embed_captions(
        self,
        cap,  # (N, 512) vector caption
    ):
        return F.normalize(self.clip_proj(cap), dim=-1) # (N, 512)



# raw (512, 256) level 0, not normalized

def span_mean(
    raw,     # (D, T) feature tầng 0 của một video, chưa chuẩn hóa
    length,  # số bước thật (không tính padding) của video
    segs,    # (N, 2) các đoạn (đầu, cuối), đơn vị là bước của tầng 0
):
    """Trung bình feature trên từng đoạn.

    Dùng tổng cộng dồn (cumsum) nên không phải lặp qua từng đoạn. Đoạn ngắn hơn một bước
    thì vẫn lấy ít nhất một bước.
    """
    cs = F.pad(raw.cumsum(-1), (1, 0))                         # (D, T+1)
    lo = segs[:, 0].floor().clamp(min=0, max=length - 1).long()
    hi = segs[:, 1].ceil().long() + 1
    hi = torch.maximum(hi.clamp(max=length), lo + 1)
    return (cs[:, hi] - cs[:, lo]).t() / (hi - lo).unsqueeze(1).float()   # (N, D)


class SegmentContext(nn.Module):
    """Vector của một đoạn, có thêm ngữ cảnh của cả video.

    Đầu vào gồm trung bình trên đoạn, trung bình trên cả video (đang nấu món gì) và vị trí
    tương đối của đoạn (bước đầu hay bước cuối của công thức). Lớp cuối khởi tạo bằng 0,
    nên lúc đầu kết quả đúng bằng trung bình trên đoạn.
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
        raw,     # (D, T) feature tầng 0 của một video
        length,  # số bước thật của video
        segs,    # (N, 2) các đoạn, đơn vị bước
    ):
        span = span_mean(raw, length, segs)                          # (N, D)
        glob = raw[:, :length].mean(-1).expand_as(span)              # (N, D)
        pos = (segs / float(length)).clamp(0, 1)                     # (N, 2)
        return F.normalize(span + self.mlp(torch.cat((span, glob, pos), -1)), dim=-1)


class BoundaryHead(nn.Module):
    """Ở mỗi bước: khoảng cách tới điểm đầu và điểm cuối của event chứa nó (tính theo stride của tầng,
    luôn >= 0), và một logit IoU-quality (đoạn dự đoán từ bước này tốt cỡ nào, dùng để xếp hạng)."""

    def __init__(
        self,
        in_dim,          # số kênh của feature pyramid đầu vào
        feat_dim,        # số kênh bên trong head
        fpn_levels=6,    # số tầng pyramid: mỗi tầng có một hệ số scale riêng cho khoảng cách
        n_layers=3,      # số lớp conv
        ks=3,            # kernel size
        with_ln=True,    # LayerNorm sau mỗi conv
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.scale = nn.ModuleList([Scale() for _ in range(fpn_levels)])
        self.out = MaskedConv1D(feat_dim, 2, ks, stride=1, padding=ks // 2)
        torch.nn.init.constant_(self.out.conv.bias, 1.0)
        # đoán IoU giữa đoạn dự đoán từ bước này và đoạn GT, dùng để xếp hạng
        self.iou_out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)

    def forward(
        self,
        fpn_feats,  # list 6 tầng, mỗi tầng (B, C, T_l)
        fpn_masks,  # list 6 tầng, mỗi tầng (B, 1, T_l) kiểu bool
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
    """Tách các bước của video rồi gán caption cho từng bước. Không có lớp phân loại."""

    def __init__(
        self,
        backbone_arch,       # (#conv, #transformer ở stem, #transformer ở nhánh pyramid), ví dụ (2, 3, 5)
        scale_factor,        # hệ số giảm độ dài giữa hai tầng pyramid
        input_dim_V,         # số kênh feature video (768)
        input_dim_A,         # số kênh feature audio (768)
        n_head,              # số head attention trong backbone
        embd_kernel_size,    # kernel size của các conv embedding trong backbone
        embd_dim,            # số kênh mỗi luồng của backbone (512); các head đọc cả hai luồng: 2 * embd_dim
        embd_with_ln,        # LayerNorm sau các conv embedding
        head_dim,            # số kênh trong mọi head, cũng là kích thước D của vector bước / đoạn
        omni_head_dim,
        regression_range,    # mỗi tầng phụ trách những đoạn có khoảng cách xa nhất tới biên nằm trong khoảng này
        head_num_layers,     # số lớp conv của mỗi head
        head_kernel_size,    # kernel size của các head
        head_with_ln,        # LayerNorm trong các head
        use_abs_pe,          # cộng position embedding dạng sin trong backbone
        train_cfg,           # trọng số các loss, dropout, droppath, cls_prior_prob
        test_cfg,            # thiết lập lúc suy luận: ngưỡng, số đoạn tối đa, NMS
        max_seq_len,         # số bước của mọi video (256)
        nmax=16,             # số caption tối đa trong một video
        clip_dim=512,        # kích thước vector caption,
        omi_feats_dim=3584
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
                                    head_with_ln, clip_dim, omi_feats_dim)
        self.seg_ctx = SegmentContext(head_dim)
        
    def set_caption_pool(
        self,
        pool_raw,  # (N, clip_dim) vector InternVideo2 của các caption khác nhau trong tập train
        omni_pool_raw
    ):
        self.pool_raw = pool_raw # pool_emb
        self.omni_pool_raw = omni_pool_raw

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

    @torch.no_grad()
    def preprocessing(
        self,
        video_list,        # list B dict, mỗi dict là một video (YouCook2Dataset.__getitem__)
        padding_val=0.0,   # giá trị điền vào các bước padding
    ):
        """Gộp các video trong batch thành tensor (B, C=768, T) và mask (B, 1, T).
        T luôn bằng max_seq_len: video bị cắt lúc train sẽ được padding cho đủ."""

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
        
        #       (B, 1, T)
        len0 = msk[0].squeeze(1).sum(-1)                                 # (B,)
        #                         (B, T)
        fpn_masks = torch.cat([m.squeeze(1) for m in msk], dim=1)        # (B, P)
        self._fpn = (feats, msk, em)                                     # để ground head dùng lại
        if self.training:
            return self.losses(video_list, fpn_masks, ev, bd, qu, em, omni_em, raw0, len0)
        return self.inference(video_list, fpn_masks, ev, bd, qu, em, raw0, len0)

    def ground(
        self,
        vid,    # (Q,) chỉ số trong batch của video mà mỗi câu hỏi tới
        query,  # (Q, 512) vector caption
    ):
        """Logit (Q, P) của ground head, dùng feature của lần forward gần nhất."""
        feats, msk, em = self._fpn
        return self.ground_head(feats, msk, vid, query, self.embed_head.embed_captions(query), em)

    def losses(
        self,
        video_list,  # list B dict, mỗi dict là một video
        valid,       # (B, P) bool, True ở các điểm thuộc bước thật (không phải padding)
        ev,          # (B, P) logit của event head
        bd,          # (B, P, 2) khoảng cách dự đoán tới điểm đầu / cuối, tính theo stride
        qu,          # (B, P) logit IoU-quality
        em,          # (B, P, D) embedding đã chuẩn hóa của mọi bước
        omni_em,     # (B, P, D)
        raw0,        # (B, D, T) feature tầng 0 của embedding head, chưa chuẩn hóa (để lấy trung bình đoạn)
        len0,        # (B,) số bước thật ở tầng 0 của mỗi video
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
        pool_o = self.omni_embed_head.embed_captions(self.omni_pool_raw)
        scale_o = self.omni_embed_head.logit_scale.exp().clamp(max=100)
        omni_emb_loss = F.cross_entropy(scale_o * omni_em[ri, ci] @ pool_o.t(), gt)
        
        
        # 5. embedding cả đoạn: vector của mỗi đoạn GT phải chọn đúng caption của nó trong cả pool
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
        video_list,  # list B dict, mỗi dict là một video
        valid,       # (B, P) bool, True ở các điểm thuộc bước thật
        ev,          # (B, P) logit của event head
        bd,          # (B, P, 2) khoảng cách dự đoán tới điểm đầu / cuối, tính theo stride
        qu,          # (B, P) logit IoU-quality
        em,          # (B, P, D) embedding đã chuẩn hóa của mọi bước
        raw0,        # (B, D, T) feature tầng 0 của embedding head (để tính vector đoạn)
        len0,        # (B,) số bước thật ở tầng 0 của mỗi video
    ):
        """Điểm của một đoạn = điểm event nhân điểm IoU-quality. Trả về mỗi video một dict:
        segments (K, 2) tính bằng giây, xếp từ tốt đến kém; embeds (K, D) vector của các đoạn;
        gt_segments, gt_text là các bước GT (giây) và caption của chúng.
        """
        t = self.test_cfg
        a = t['iou_power']
        prob = ev.sigmoid().pow(1 - a) * qu.sigmoid().pow(a) * valid.float() # (B, P)
        out = []
        for b, v in enumerate(video_list):
            pts = torch.cat(v['points'], dim=0).to(ev.device)   # (P, 4): t, cận dưới, cận trên của khoảng hồi quy, stride
            #          (P,)
            keep = (prob[b] > t['score_thresh']).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                keep = prob[b].topk(1).indices
            sc, order = prob[b][keep].topk(min(t['max_seg_num'] * 5, keep.numel()))
            keep = keep[order]
            # đoạn của một điểm: từ t - (khoảng cách tới đầu) * stride tới t + (khoảng cách tới cuối) * stride
            p, o = pts[keep], bd[b][keep]
            segs = torch.stack((p[:, 0] - o[:, 0] * p[:, 3], p[:, 0] + o[:, 1] * p[:, 3]), -1).cpu()
            scores = sc.cpu()
            ok = (segs[:, 1] - segs[:, 0]) > t['duration_thresh']
            segs, scores = segs[ok], scores[ok]
            # soft-NMS không phân lớp: mọi đoạn chung một nhãn giả
            segs, scores, _ = batched_nms(segs, scores, torch.arange(len(segs)), 0.7, 0.001,
                                          t['max_seg_num'], use_soft_nms=True, sigma=t['nms_sigma'])
            # vector của đoạn = trung bình trên cả đoạn, tính khi đoạn còn ở đơn vị bước
            span = self.seg_ctx(raw0[b], int(len0[b]), segs.to(raw0.device)).cpu()

            # đổi từ đơn vị bước sang giây (xem __getitem__ của dataset)
            st, nf, fps = v['feat_stride'], v['feat_num_frames'], v['fps']
            segs = ((segs * st + 0.5 * nf) / fps).clamp(min=0.0, max=float(v['duration']))
            out.append({'video_id': v['video_id'], 'segments': segs, 'scores': scores, 'embeds': span,
                        'gt_segments': v['segments_sec'], 'gt_text': v['cap_text']})
        return out
