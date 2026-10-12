"""Run a training checkpoint on YouCook2 val and dump every prediction (all 100 after NMS) with the GT."""
import json, os, pickle, sys
import numpy as np, torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import train
from libs.modeling.blocks import upgrade_state_dict
from libs.modeling.event_archs import EventCaptionTransformer

ckpt, run, out = sys.argv[1:4]
if not hasattr(np, '_core'):   # checkpoint pickled with numpy 2 on Colab
    sys.modules.setdefault('numpy._core', np.core); sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
cfg = yaml.safe_load(open('configs/youcook2_event.yaml'))
cfg = train_cfg = json.load(open(run))['config']
cfg['coin']['train'] = False
for k in cfg.get('extra_train', {}):
    cfg['extra_train'][k]['ratio'] = 0.0
cfg['num_workers'] = 4
rng = train.fix_random_seed(1, include_cuda=True)
ds, dv, dl, dlv, extra, ov = train.build_loaders(cfg, rng)
sd = upgrade_state_dict({k: v.float() for k, v in train.load_ckpt(ckpt)['state_dict'].items()})
mcfg = dict(cfg['model'], input_dim_V=ds.feat_dims[0], input_dim_A=ds.feat_dims[1],
            omni_dim=sd['omni_proj.2.weight'].shape[0] if 'omni_proj.2.weight' in sd else 0)
dev = torch.device('cuda')
model = EventCaptionTransformer(**mcfg).to(dev)
model.load_state_dict(sd, strict=False)
model.eval()
res = []
pool = torch.nn.functional.normalize(model.embed_head.embed_captions(torch.from_numpy(ds.pool_train).to(dev).float()), dim=-1)
scale = float(model.embed_head.logit_scale.exp().clamp(max=100))
print('pool', tuple(pool.shape), 'scale %.1f' % scale)
with torch.no_grad():
    for batch in dlv:
        for x in batch:
            x['feats'] = {k: v.to(dev) for k, v in x['feats'].items()}
        for r, x in zip(model(batch), batch):
            q = torch.nn.functional.normalize(r['embeds'].to(dev).float(), dim=-1)
            sim = q @ pool.t()                                     # (K, pool)
            top = sim.topk(20, dim=-1).values
            match = {'max': top[:, 0].cpu().numpy(), 'top20': top.mean(-1).cpu().numpy(),
                     'lse': (torch.logsumexp(scale * sim, -1) / scale).cpu().numpy()}
            res.append({'match': match, 'vid': r['video_id'], 'dur': float(x['duration']), 'pred': r['segments'].numpy(),
                        'score': r['scores'].numpy(), 'gt': np.asarray(r['gt_segments'], float)})
pickle.dump(res, open(out, 'wb'))
print(len(res), 'videos')
