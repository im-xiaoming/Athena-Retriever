"""Huấn luyện mô hình tách sự kiện và sinh mô tả.

  python train_event.py configs/youcook2_event.yaml --output run1
  python train_event.py configs/youcook2_event.yaml --eval ckpt/run1/best_cap.pth.tar

--pretrain nạp checkpoint UniAV gốc, nhưng đo hai lần đều kém hơn train từ đầu.
"""
import argparse, os, time, yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from libs.datasets import make_dataset
from libs.datasets.data_utils import trivial_batch_collator, worker_init_reset_seed
from libs.modeling import make_multimodal_meta_arch
from libs.modeling.blocks import MaskedConv1D, LayerNorm, Scale, AffineDropPath
from libs.utils import make_scheduler, fix_random_seed, ModelEma


def build_optimizer(model, opt):
    decay, no_decay, clip = set(), set(), set()
    white = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    black = (LayerNorm, torch.nn.GroupNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if 'clip_proj' in fpn:
                clip.add(fpn)
            elif pn.endswith('weight') and isinstance(m, white):
                decay.add(fpn)
            else:
                no_decay.add(fpn)
    pd = dict(model.named_parameters())
    assert not (pd.keys() - (decay | no_decay | clip))
    return torch.optim.AdamW([
        {'params': [pd[n] for n in sorted(decay)], 'weight_decay': opt['weight_decay']},
        {'params': [pd[n] for n in sorted(no_decay)], 'weight_decay': 0.0},
        {'params': [pd[n] for n in sorted(clip)], 'weight_decay': 0.0, 'lr': opt['clip_proj_lr']},
    ], lr=opt['learning_rate'])


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def mbr_pick(s, pool_n, scale, k=20):
    """Chọn câu theo đồng thuận thay vì lấy câu điểm cao nhất.

    s (N,) là điểm của từng câu trong kho. Lấy k câu điểm cao nhất, đổi điểm thành
    xác suất, rồi chọn câu giống nhất với cả nhóm (minimum Bayes risk). Câu được
    nhiều ứng viên mạnh cùng ủng hộ thắng câu chỉ tình cờ đứng đầu.
    """
    top = s.topk(k)
    p = F.softmax(scale * top.values, dim=0)
    cand = pool_n[top.indices]
    agree = (cand @ cand.t()) @ p
    return int(top.indices[0]), int(top.indices[agree.argmax()])


@torch.no_grad()
def evaluate(model, loader, device, pool_raw, pool_text, ths=(0.3, 0.5, 0.7), full=False):
    """Chấm hai việc: tách đoạn có đúng không, và mô tả có đúng không.

    Phần mô tả chỉ chấm trên các GT có đoạn dự đoán khớp IoU >= 0.3.
    cos_gt   : cosine giữa vector đoạn dự đoán và caption THẬT của đoạn đó
    cos_rand : cùng phép đo nhưng với một caption ngẫu nhiên, đây là mức sàn
    ret_sim  : câu chọn ra giống câu thật tới đâu, trong không gian ONE-PEACE
    CIDEr    : thước đo chuẩn của bài toán sinh mô tả, so chữ với câu thật,
               độc lập hoàn toàn với ONE-PEACE
    full     : chấm thêm các biến thể (câu đứng đầu, vector tại một mốc, từng không
               gian riêng khi có thầy OmniRetriever) để so sánh
    Khi có thầy, điểm của một câu là tổng cosine trong hai không gian.
    """
    from pycocoevalcap.cider.cider import Cider
    model.eval()
    pool_f = F.normalize(model.embed_head.clip_proj(pool_raw), dim=-1)
    pool_n = F.normalize(pool_raw, dim=-1)
    scale = float(model.embed_head.logit_scale.exp().clamp(max=100))
    omni = model.omni_dim > 0
    if omni:
        o_txt = model.omni_text.masked_fill(~model.omni_text_ok[:, None], 0)
    cov = {t: 0 for t in ths}
    n_gt = n_pred = 0
    best_ious, cos_gt, cos_rand = [], [], []
    ret = {v: [] for v in ('top1', 'mbr', 'pt_top1', 'mbr_onepeace', 'mbr_omni', 'oracle')}
    gts_txt, hyp = {}, {v: {} for v in ret}
    gts_or, hyp_or = {}, {}
    g = torch.Generator().manual_seed(0)
    for batch in loader:
        for x in batch:
            x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
        res = model(batch)
        for r, x in zip(res, batch):
            gts, texts = r['gt_segments'], r['gt_text']
            k = min(len(gts), r['segments'].shape[0])
            preds = r['segments'][:k].tolist()
            emb = r['embeds'][:k].to(device).float()
            emb_pt = r['embeds_pt'][:k].to(device).float()
            n_pred += k
            cap_raw = x['cap_emb'].to(device).float()
            cap_proj = F.normalize(model.embed_head.clip_proj(cap_raw), dim=-1)
            cap_n = F.normalize(cap_raw, dim=-1)
            if full:
                # mo ta tren chinh doan GT: chi do phan mo ta, khong phu thuoc tach doan
                for gi in range(len(gts)):
                    s = pool_f @ r['embeds_gt'][gi].to(device).float()
                    if omni:
                        s = s + o_txt @ r['embeds_gt_omni'][gi].to(device).float()
                    i = mbr_pick(s, pool_n, scale)[1]
                    ret['oracle'].append(float(pool_n[i] @ cap_n[gi]))
                    okey = len(gts_or); gts_or[okey] = [texts[gi]]; hyp_or[okey] = [pool_text[i]]
            for gi, gt in enumerate(gts):
                n_gt += 1
                if not preds:
                    best_ious.append(0.0); continue
                vals = [iou(p, gt) for p in preds]
                j = int(np.argmax(vals)); best = vals[j]
                best_ious.append(best)
                for t in ths:
                    cov[t] += (best >= t)
                if best < 0.3:
                    continue
                q = F.normalize(emb[j], dim=-1)
                cos_gt.append(float(q @ cap_proj[gi]))
                ridx = int(torch.randint(len(pool_f), (1,), generator=g))
                cos_rand.append(float(q @ pool_f[ridx]))
                key = len(gts_txt); gts_txt[key] = [texts[gi]]
                s_op = pool_f @ q
                s = s_op
                if omni:
                    s_om = o_txt @ r['embeds_omni'][j].to(device).float()
                    s = s_op + s_om
                t1, mb = mbr_pick(s, pool_n, scale)
                picks = {'top1': t1, 'mbr': mb}
                if full:
                    picks['pt_top1'] = int((pool_f @ F.normalize(emb_pt[j], dim=-1)).argmax())
                    if omni:
                        picks['mbr_onepeace'] = mbr_pick(s_op, pool_n, scale)[1]
                        picks['mbr_omni'] = mbr_pick(s_om, pool_n, scale)[1]
                for v, i in picks.items():
                    ret[v].append(float(pool_n[i] @ cap_n[gi]))
                    hyp[v][key] = [pool_text[i]]
    model.train()
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    cider = lambda v: Cider().compute_score(gts_txt, hyp[v])[0] * 100 if hyp[v] else 0.0
    out = {'R@%.1f' % t: 100.0 * cov[t] / max(n_gt, 1) for t in ths}
    out['mIoU'] = 100.0 * mean(best_ious)
    out['seg/vid'] = n_pred / max(len(loader.dataset), 1)
    out['cos_gt'] = mean(cos_gt)
    out['cos_rand'] = mean(cos_rand)
    out['ret_sim'] = mean(ret['mbr'])
    out['CIDEr'] = cider('mbr')
    if full:
        from pycocoevalcap.meteor.meteor import Meteor
        for v in ('top1', 'pt_top1') + (('mbr_onepeace', 'mbr_omni') if omni else ()):
            out['ret_sim[%s]' % v] = mean(ret[v])
            out['CIDEr[%s]' % v] = cider(v)
        out['METEOR'] = Meteor().compute_score(gts_txt, hyp['mbr'])[0] * 100
        out['ret_sim[oracle]'] = mean(ret['oracle'])
        out['CIDEr[oracle]'] = Cider().compute_score(gts_or, hyp_or)[0] * 100
        out['n_captioned'] = len(gts_txt)
    return out


def load_pretrain(model, path):
    """Nạp checkpoint UniAV gốc, ánh xạ head cũ sang head mới.

    backbone                   -> backbone
    reg_head (offset TASK1)    -> bound_head    TASK1 là ActivityNet, 2 kênh
    cls_head                   -> embed_head    cùng clip_proj, vis_proj
    cls_head (tower)           -> event_head    chỉ thân, lớp out học từ đầu
    """
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('state_dict_ema') or sd['state_dict']
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    rules = [('backbone.', 'backbone.'),
             ('reg_head.offset_head.TASK1.', 'bound_head.out.'),
             ('reg_head.', 'bound_head.'),
             ('cls_head.', 'embed_head.'),
             ('cls_head.head.', 'event_head.head.'),
             ('cls_head.norm.', 'event_head.norm.')]
    own, new = model.state_dict(), {}
    for k, v in sd.items():
        for src, dst in rules:
            if k.startswith(src):
                t = dst + k[len(src):]
                if t in own and own[t].shape == v.shape:
                    new[t] = v
    model.load_state_dict(new, strict=False)
    fresh = [k for k in own if k not in new]
    n_new = sum(own[k].numel() for k in new) / 1e6
    n_all = sum(v.numel() for v in own.values()) / 1e6
    print('Pretrain     : %s' % path)
    print('               loaded %.1fM / %.1fM params, from scratch: %s'
          % (n_new, n_all, ', '.join(fresh) or 'none'), flush=True)


# Bảng log một dòng mỗi epoch. Mỗi cột: (tên in ra, khoá, độ rộng, định dạng)
GROUPS = [
    ('train loss', [('event', 'ev_loss', 6, '.3f'), ('reg', 'reg_loss', 6, '.3f'), ('iou', 'iou_loss', 6, '.3f'),
                    ('embed', 'emb_loss', 6, '.3f'), ('span', 'span_loss', 6, '.3f'),
                    ('total', 'final_loss', 6, '.3f')]),
    ('segmentation (val)', [('R@0.3', 'R@0.3', 6, '.2f'), ('R@0.5', 'R@0.5', 6, '.2f'),
                            ('R@0.7', 'R@0.7', 6, '.2f'), ('mIoU', 'mIoU', 6, '.2f'),
                            ('seg/vid', 'seg/vid', 7, '.2f')]),
    ('captioning (val)', [('cos_gt', 'cos_gt', 6, '.3f'), ('cos_rand', 'cos_rand', 8, '.3f'),
                          ('ret_sim', 'ret_sim', 7, '.3f'), ('CIDEr', 'CIDEr', 6, '.2f')]),
]
LEAD = '{:>6} {:>5} {:>8}'


def use_omni_columns():
    """Thêm hai cột loss của thầy OmniRetriever vào bảng."""
    global HEAD, RULE
    GROUPS[0][1][5:5] = [('o_txt', 'omni_txt_loss', 6, '.3f'), ('o_av', 'omni_av_loss', 6, '.3f')]
    HEAD = LEAD.format('epoch', 'time', 'lr') + ' | ' + _cells(lambda c: '%*s' % (c[2], c[0]))
    RULE = '-' * (len(HEAD) + 4)


def _cells(fn):
    return ' | '.join(' '.join(fn(c) for c in cols) for _, cols in GROUPS)


HEAD = LEAD.format('epoch', 'time', 'lr') + ' | ' + _cells(lambda c: '%*s' % (c[2], c[0]))
RULE = '-' * (len(HEAD) + 4)


def print_header():
    print('Legend       : S = new best R@0.5, C = new best CIDEr (each saves its checkpoint)')
    print('               cos_rand = cosine to a random caption (floor), ret_sim = similarity of')
    print('               chosen caption to GT caption (ONE-PEACE space), CIDEr = word overlap')
    print(RULE)
    widths = [sum(c[2] for c in cols) + len(cols) - 1 for _, cols in GROUPS]
    print(LEAD.format('', '', '') + ' | ' + ' | '.join(
        '{:^{}}'.format(name, w) for (name, _), w in zip(GROUPS, widths)))
    print(HEAD)
    print(RULE, flush=True)


def print_row(ep, t, lr, vals, mark):
    row = LEAD.format(ep, '%.0fs' % t, '%.1e' % lr) + ' | ' + _cells(
        lambda c: format(vals.get(c[1], float('nan')), '>%d%s' % (c[2], c[3])))
    print(row + ('  ' + mark if mark else ''), flush=True)


def build_loaders(cfg, rng):
    ds = make_dataset('youcook2_cap', True, cfg['train_split'], **cfg['dataset'])
    dv = make_dataset('youcook2_cap', False, cfg['val_split'], **cfg['dataset'])
    dl = DataLoader(ds, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                    sampler=RandomSampler(ds), collate_fn=trivial_batch_collator,
                    worker_init_fn=worker_init_reset_seed, drop_last=True, generator=rng,
                    persistent_workers=cfg['num_workers'] > 0)
    dlv = DataLoader(dv, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                     sampler=SequentialSampler(dv), collate_fn=trivial_batch_collator)
    return ds, dv, dl, dlv


def eval_only(a, cfg, model, dlv, dev, pool_raw, pool_text):
    """Chấm một checkpoint với nhiều cấu hình NMS, in đủ các biến thể."""
    ck = torch.load(a.eval, map_location='cpu')
    model.load_state_dict(ck['state_dict'])
    print('Checkpoint   : %s (epoch %d)' % (a.eval, ck['epoch'] + 1), flush=True)
    for power in a.iou_power.split(','):
        model.iou_power = float(power)
        for spec in a.nms.split(','):
            kind, *rest = spec.split(':')
            model.nms_cfg = {'soft': kind == 'soft', 'iou': float(rest[0]),
                             'sigma': float(rest[1]) if len(rest) > 1 else 0.5}
            m = evaluate(model, dlv, dev, pool_raw, pool_text, full=not a.fast)
            print('\niou_power %s  NMS %-14s' % (power, spec)
                  + '  '.join('%s %.3f' % (k, v) for k, v in m.items()), flush=True)


def main(a):
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = fix_random_seed(cfg.get('init_rand_seed', 1234567891), include_cuda=True)
    out_dir = os.path.join(cfg['output_folder'], a.output)

    ds, dv, dl, dlv = build_loaders(cfg, rng)
    print('Data         : %d train / %d val videos' % (len(ds), len(dv)), flush=True)

    if ds.omni:
        cfg['model']['omni_dim'] = ds.omni_dim
        use_omni_columns()
    model = make_multimodal_meta_arch('EventCaptionTransformer', **cfg['model'])
    if a.pretrain:
        load_pretrain(model, a.pretrain)
    model = model.to(dev)
    print('Parameters   : %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    # kho caption lay tu TAP TRAIN, khong dung caption cua tap val
    pool_raw = torch.from_numpy(ds.pool_emb).to(dev)
    model.set_caption_pool(pool_raw)
    if ds.omni:
        model.set_omni_pool(torch.from_numpy(ds.omni_text_pool).to(dev),
                            torch.from_numpy(ds.omni_text_ok).to(dev),
                            torch.from_numpy(ds.omni_av_pool).to(dev))
    print('Caption pool : %d unique train captions' % len(ds.pool_text), flush=True)
    if a.eval:
        return eval_only(a, cfg, model, dlv, dev, pool_raw, ds.pool_text)
    os.makedirs(out_dir, exist_ok=True)
    print('Output       : %s' % out_dir, flush=True)

    opt = build_optimizer(model, cfg['opt'])
    sch = make_scheduler(opt, cfg['opt'], len(dl), a.epochs)
    n_ep = a.epochs + cfg['opt']['warmup_epochs']
    best = {'R@0.5': (-1.0, 0, 'best_seg'), 'CIDEr': (-1.0, 0, 'best_cap')}
    print_header()
    for ep in range(n_ep):
        t0 = time.time(); acc = {}
        for batch in dl:
            for x in batch:
                x['feats'] = {k: v.to(dev) for k, v in x['feats'].items()}
            L = model(batch)
            opt.zero_grad(set_to_none=True); L['final_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train_cfg']['clip_grad_l2norm'])
            opt.step(); sch.step()
            for k, v in L.items(): acc[k] = acc.get(k, 0.0) + v.detach().item()
        m = evaluate(model, dlv, dev, pool_raw, ds.pool_text)
        mark = ''
        for key, (val, _, name) in best.items():
            if m[key] > val:
                best[key] = (m[key], ep, name); mark += 'S' if key == 'R@0.5' else 'C'
                torch.save({'epoch': ep, 'state_dict': model.state_dict(), 'metrics': m},
                           os.path.join(out_dir, name + '.pth.tar'))
        vals = dict(m, **{k: v / len(dl) for k, v in acc.items()})
        print_row('%d/%d' % (ep + 1, n_ep), time.time() - t0, sch.get_last_lr()[0], vals, mark)
    print(RULE)
    for key, (val, ep, name) in best.items():
        print('Best %-6s = %6.2f at epoch %d -> %s' % (
            key, val, ep + 1, os.path.join(out_dir, name + '.pth.tar')), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config'); p.add_argument('--output', default='ev1')
    p.add_argument('--epochs', type=int, default=10,
                   help='40 epoch overfit tu epoch 7, 10 la du')
    p.add_argument('--pretrain', default='', help='checkpoint UniAV gốc để khởi tạo')
    p.add_argument('--eval', default='', help='chỉ chấm checkpoint này, không train')
    p.add_argument('--nms', default='soft:0.7:0.5,soft:0.5:0.5,hard:0.5,hard:0.3',
                   help='các cấu hình NMS để chấm khi dùng --eval')
    p.add_argument('--iou-power', default='0.5',
                   help='trọng số điểm IoU khi xếp hạng đoạn, 0 là chỉ dùng điểm sự kiện')
    p.add_argument('--fast', action='store_true', help='--eval: bỏ các biến thể chậm')
    main(p.parse_args())
