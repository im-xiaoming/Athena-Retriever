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

import itertools
out_all = {}
for ip, sig in itertools.product((0.2, 0.3, 0.5, 0.7), (0.5, 0.9)):
    model.iou_power = ip; model.nms_cfg['sigma'] = sig
    res = []
    with torch.no_grad():
        for batch in dlv:
            for x in batch:
                x['feats'] = {k: v.to(dev) for k, v in x['feats'].items()}
            for r, x in zip(model(batch), batch):
                res.append({'vid': r['video_id'], 'dur': float(x['duration']), 'pred': r['segments'].numpy(),
                            'score': r['scores'].numpy(), 'gt': np.asarray(r['gt_segments'], float)})
    out_all[(ip, sig)] = res
    print('done', ip, sig, flush=True)
pickle.dump(out_all, open(out, 'wb'))
