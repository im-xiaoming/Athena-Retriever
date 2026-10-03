"""Huấn luyện mô hình tách sự kiện và sinh mô tả.

  python train_event.py configs/youcook2_event.yaml --output run1 --epochs 40
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


@torch.no_grad()
def evaluate(model, loader, device, pool_proj, pool_raw, pool_text, ths=(0.3, 0.5, 0.7)):
    """Chấm hai việc: tách đoạn có đúng không, và mô tả có đúng không.

    cos_gt   : cosine giữa vector đoạn dự đoán và caption THẬT của đoạn đó
    cos_nen  : cùng phép đo nhưng với một caption ngẫu nhiên, đây là mức sàn
    sim_truy : caption lấy ra từ kho giống caption thật tới đâu, đo trong
               không gian ONE-PEACE gốc
    """
    model.eval()
    cov = {t: 0 for t in ths}
    n_gt = n_pred = 0
    best_ious, cos_gt, cos_nen, sim_truy = [], [], [], []
    g = torch.Generator().manual_seed(0)
    for batch in loader:
        for x in batch:
            x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
        res = model(batch)
        for r, x in zip(res, batch):
            gts, texts = r['gt_segments'], r['gt_text']
            k = min(len(gts), r['segments'].shape[0])
            preds = r['segments'][:k].tolist()
            emb = r['embeds'][:k].to(device)
            n_pred += k
            cap_raw = x['cap_emb'].to(device).float()          # (NMAX, 1536) chua chieu
            cap_proj = F.normalize(model.embed_head.clip_proj(cap_raw), dim=-1)
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
                q = F.normalize(emb[j].float(), dim=-1)
                cos_gt.append(float(q @ cap_proj[gi]))
                ridx = int(torch.randint(len(pool_proj), (1,), generator=g))
                cos_nen.append(float(q @ pool_proj[ridx]))
                top = int((pool_proj @ q).argmax())
                sim_truy.append(float(F.normalize(pool_raw[top], dim=-1) @
                                      F.normalize(cap_raw[gi], dim=-1)))
    model.train()
    out = {'Recall@IoU%.1f' % t: 100.0 * cov[t] / max(n_gt, 1) for t in ths}
    out['mIoU'] = 100.0 * float(np.mean(best_ious)) if best_ious else 0.0
    out['cos_gt'] = float(np.mean(cos_gt)) if cos_gt else 0.0
    out['cos_nen'] = float(np.mean(cos_nen)) if cos_nen else 0.0
    out['sim_truy'] = float(np.mean(sim_truy)) if sim_truy else 0.0
    out['doan/video'] = n_pred / max(len(loader.dataset), 1)
    return out


def main(a):
    cfg = yaml.safe_load(open(a.config))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = fix_random_seed(cfg.get('init_rand_seed', 1234567891), include_cuda=True)
    out_dir = os.path.join(cfg['output_folder'], a.output); os.makedirs(out_dir, exist_ok=True)

    ds = make_dataset('youcook2_cap', True, cfg['train_split'], **cfg['dataset'])
    dv = make_dataset('youcook2_cap', False, cfg['val_split'], **cfg['dataset'])
    print('train %d | val %d video' % (len(ds), len(dv)), flush=True)
    dl = DataLoader(ds, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                    sampler=RandomSampler(ds), collate_fn=trivial_batch_collator,
                    worker_init_fn=worker_init_reset_seed, drop_last=True, generator=rng,
                    persistent_workers=cfg['num_workers'] > 0)
    dlv = DataLoader(dv, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                     sampler=SequentialSampler(dv), collate_fn=trivial_batch_collator)

    model = make_multimodal_meta_arch('EventCaptionTransformer', **cfg['model']).to(dev)
    print('tham so %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    # kho caption lay tu TAP TRAIN, khong dung caption cua tap val
    seen, pool_raw, pool_text = set(), [], []
    for it in ds.data_list:
        for i in range(it['n_cap']):
            k = '%s#%d' % (it['id'], i); t = str(ds.cap_text[k])
            if t in seen: continue
            seen.add(t); pool_raw.append(ds.cap_emb[k]); pool_text.append(t)
    pool_raw = torch.from_numpy(np.stack(pool_raw).astype(np.float32)).to(dev)
    print('kho caption: %d cau' % len(pool_text), flush=True)

    opt = build_optimizer(model, cfg['opt'])
    sch = make_scheduler(opt, cfg['opt'], len(dl), a.epochs)
    best = -1.0
    for ep in range(a.epochs + cfg['opt']['warmup_epochs']):
        t0 = time.time(); acc = {}
        for batch in dl:
            for x in batch:
                x['feats'] = {k: v.to(dev) for k, v in x['feats'].items()}
            L = model(batch)
            opt.zero_grad(set_to_none=True); L['final_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train_cfg']['clip_grad_l2norm'])
            opt.step(); sch.step()
            for k, v in L.items(): acc[k] = acc.get(k, 0.0) + float(v)
        with torch.no_grad():
            pool = F.normalize(model.embed_head.clip_proj(pool_raw), dim=-1)
        m = evaluate(model, dlv, dev, pool, pool_raw, pool_text)
        print('ep %2d | %s | %.0fs | %s' % (
            ep, ' '.join('%s %.3f' % (k, v/len(dl)) for k, v in acc.items()),
            time.time()-t0, ' '.join('%s %.2f' % (k, v) for k, v in m.items())), flush=True)
        if m['Recall@IoU0.5'] > best:
            best = m['Recall@IoU0.5']
            torch.save({'epoch': ep, 'state_dict': model.state_dict(), 'metrics': m},
                       os.path.join(out_dir, 'best.pth.tar'))
            print('   -> luu best', flush=True)
    print('xong, best Recall@IoU0.5 = %.2f' % best, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config'); p.add_argument('--output', default='ev1')
    p.add_argument('--epochs', type=int, default=40)
    main(p.parse_args())
