"""Ensemble of segmentation checkpoints on YouCook2 val: per-point mean of event / IoU probabilities and boundary
offsets (same 504-point grid), then the usual scoring + soft-NMS; optional segment voting (UniAV/ActionFormer)."""
import json, os, pickle, sys, itertools
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if not os.path.exists(os.path.join(ROOT, 'train.py')): ROOT = '/home/minh/projects/UniAV'
sys.path.insert(0, ROOT); os.chdir(ROOT)
import train
from torch.utils.data import DataLoader, SequentialSampler
from libs.datasets.youcook2_cap import YouCook2CaptionDataset
from libs.datasets.data_utils import trivial_batch_collator
from libs.modeling.blocks import upgrade_state_dict
from libs.modeling.event_archs import EventCaptionTransformer
from libs.utils import batched_nms

MEMBERS = {   # name: (checkpoint, config source)
    'E1': ('ckpt/api/athena.pth', None), 'R1': ('ckpt/api/athena_coin.pth', None),
    'all_b16': ('ckpt/all_b16/best_seg.pth.tar', 'experiments/runs/b8a86a5e6ef7-all_b16.json'),
    'ft_yc2': ('ckpt/ft_yc2/best_seg.pth.tar', 'experiments/runs/DESKTOP-8PSQBN9-ft_yc2.json'),
    'ft_other': ('ckpt/ft_other/best_seg.pth.tar', 'experiments/runs/DESKTOP-8PSQBN9-ft_other.json')}
dev = torch.device('cuda')

def merge(base, over):
    for k, v in over.items():
        base[k] = merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base

def load(ck_path, run):
    import yaml
    cfg = yaml.safe_load(open('configs/youcook2_event.yaml'))
    ck = train.load_ckpt(ck_path)
    if run is None:
        cfg, sd = merge(cfg, ck['config']), ck.get('state_dict_seg', ck['state_dict'])
    else:
        cfg, sd = merge(cfg, json.load(open(run))['config']), ck['state_dict']
    sd = upgrade_state_dict({k: v.float() for k, v in sd.items()})
    m = dict(cfg['model'], omni_dim=sd['omni_proj.2.weight'].shape[0] if 'omni_proj.2.weight' in sd else 0)
    m['input_dim_V'] = sd['backbone.embd_V.0.conv.weight'].shape[1] if 'backbone.embd_V.0.conv.weight' in sd else m.get('input_dim_V', 768)
    m['input_dim_A'] = sd['backbone.embd_A.0.conv.weight'].shape[1] if 'backbone.embd_A.0.conv.weight' in sd else m.get('input_dim_A', 768)
    model = EventCaptionTransformer(**m).to(dev)
    miss, unused = model.load_state_dict(sd, strict=False)
    assert not [k for k in miss if not k.startswith(('ground_head', 'omni'))], miss
    model.eval(); model.w_ground = 0
    return model, cfg

def heads(model, batch):
    """event / IoU probabilities and offsets per point, from a patched inference"""
    got = {}
    def cap(video_list, valid, ev, bd, qu, em, raw0, len0):
        got.update(ev=ev.sigmoid(), qu=qu.sigmoid(), bd=bd, valid=valid); return []
    model.inference = cap
    model(batch)
    return got

def segments(v, ev, qu, bd, valid, a=0.3, vote=None, sigma=0.5, thresh=0.05, max_seg=100):
    prob = ev.pow(1 - a) * qu.pow(a) * valid.float()
    pts = torch.cat(v['points'], dim=0).to(ev.device)
    keep = (prob > thresh).nonzero(as_tuple=True)[0]
    if keep.numel() == 0: keep = prob.topk(1).indices
    sc, order = prob[keep].topk(min(max_seg * 5, keep.numel())); keep = keep[order]
    p, o = pts[keep], bd[keep]
    segs = torch.stack((p[:, 0] - o[:, 0] * p[:, 3], p[:, 0] + o[:, 1] * p[:, 3]), -1).cpu(); sc = sc.cpu()
    ok = (segs[:, 1] - segs[:, 0]) > 0.05; segs, sc = segs[ok], sc[ok]
    allsegs, allsc = segs.clone(), sc.clone()
    s2, sc2, _ = batched_nms(segs, sc, torch.arange(len(segs)), 0.7, 0.001, max_seg, use_soft_nms=True, sigma=sigma)
    if vote is not None and len(s2):   # UniAV seg_voting: refine each kept segment by its overlapping candidates
        inter = (torch.minimum(s2[:, None, 1], allsegs[None, :, 1]) - torch.maximum(s2[:, None, 0], allsegs[None, :, 0])).clamp(min=0)
        iou = inter / ((s2[:, 1] - s2[:, 0])[:, None] + (allsegs[:, 1] - allsegs[:, 0])[None] - inter)
        w = (iou >= vote).float() * (allsc + 1.5)[None]
        s2 = (w / w.sum(1, keepdim=True)) @ allsegs
    st, nf, fps = v['feat_stride'], v['feat_num_frames'], v['fps']
    return ((s2 * st + 0.5 * nf) / fps).clamp(0, float(v['duration'])).numpy()

if __name__ == '__main__':
    out = sys.argv[1]
    models = {n: load(*MEMBERS[n]) for n in MEMBERS}
    cfg = models['all_b16'][1]
    d = YouCook2CaptionDataset(False, cfg['val_split'], **cfg['dataset'])
    dl = DataLoader(d, batch_size=8, num_workers=4, sampler=SequentialSampler(d), collate_fn=trivial_batch_collator)
    store = []   # per video: probabilities of every member, for offline combination
    with torch.no_grad():
        for batch in dl:
            for x in batch:
                x['feats'] = {k: v.to(dev) for k, v in x['feats'].items()}
            hs = {n: heads(m, batch) for n, (m, _) in models.items()}
            for b, x in enumerate(batch):
                store.append({'v': {k: x[k] for k in ('points', 'feat_stride', 'feat_num_frames', 'fps', 'duration')},
                              'gt': np.asarray(x['segments_sec'], float),
                              'h': {n: {k: h[k][b].cpu() for k in ('ev', 'qu', 'bd', 'valid')} for n, h in hs.items()}})
    pickle.dump(store, open(out, 'wb'))
    print(len(store), 'videos')
