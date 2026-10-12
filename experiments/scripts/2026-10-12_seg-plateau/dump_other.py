"""Run a training checkpoint on a YouCook2 split (no crop) and save its top-100 proposals per video:
segments (s), scores, segment vectors q, point vectors, GT segments, duration."""
import json, os, pickle, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import train
from torch.utils.data import DataLoader, SequentialSampler
from libs.datasets.youcook2_cap import YouCook2CaptionDataset
from libs.datasets.data_utils import trivial_batch_collator
from libs.modeling.blocks import upgrade_state_dict
from libs.modeling.event_archs import EventCaptionTransformer

ckpt, run, split, out = sys.argv[1:5]
cfg = json.load(open(run))['config']
d = YouCook2CaptionDataset(False, cfg['train_split'] if split == 'train' else cfg['val_split'], **cfg['dataset'])
dl = DataLoader(d, batch_size=8, num_workers=4, sampler=SequentialSampler(d), collate_fn=trivial_batch_collator)
sd = upgrade_state_dict({k: v.float() for k, v in train.load_ckpt(ckpt)['state_dict'].items()})
mcfg = dict(cfg['model'], input_dim_V=d.feat_dims[0], input_dim_A=d.feat_dims[1],
            omni_dim=sd['omni_proj.2.weight'].shape[0] if 'omni_proj.2.weight' in sd else 0)
dev = torch.device('cuda')
model = EventCaptionTransformer(**mcfg).to(dev)
print(model.load_state_dict(sd, strict=False))
model.eval(); model.w_ground = 0   # no ground passes needed
res = []
pool = torch.nn.functional.normalize(model.embed_head.embed_captions(torch.from_numpy(d.pool_train).to(dev).float()), dim=-1)
oth = torch.nn.functional.normalize(model.embed_head.embed_captions(torch.from_numpy(d.other).to(dev).float()[None]), dim=-1)
cols = torch.cat((pool, oth)); scale = float(model.embed_head.logit_scale.exp().clamp(max=100))
def p_other(v):
    v = torch.nn.functional.normalize(v.to(dev).float(), dim=-1)
    return (scale * v @ cols.t()).softmax(-1)[:, -1].cpu().numpy()
with torch.no_grad():
    for batch in dl:
        for x in batch:
            x['feats'] = {k: v.to(dev) for k, v in x['feats'].items()}
        for r, x in zip(model(batch), batch):
            res.append({'vid': r['video_id'], 'dur': float(x['duration']), 'pred': r['segments'].numpy(),
                        'score': r['scores'].numpy(), 'q': r['embeds'].numpy().astype(np.float16),
                        'po_pt': p_other(r['embeds_pt']), 'po_q': p_other(r['embeds']),
                        'qpt': r['embeds_pt'].numpy().astype(np.float16), 'gt': np.asarray(r['gt_segments'], float)})
pickle.dump(res, open(out, 'wb'))
print(split, len(res), 'videos')
