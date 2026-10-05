"""Pack a training run into one self-describing checkpoint for uniav_api.

  python tools/export_api_ckpt.py iv2                # -> ckpt/api/uniav_iv2.pth

Keeps the best captioning checkpoint (best_cap) in fp16, without the OmniRetriever projection
(training only), plus the run's dataset / model config and its final evaluation, so the API
knows which features the model expects without configs/youcook2_event.yaml. A model trained in
the teacher's text space (caption_space omni) also gets those caption vectors (caption_pool, in
the order of uniav_api/assets/caption_pool.npz).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from libs.modeling.blocks import upgrade_state_dict  # noqa: E402
DATASET_KEYS = ('feat_source', 'default_fps', 'max_seq_len', 'force_upsampling', 'iv2_video_keys', 'iv2_l2norm',
                'iv2_rows_per_sec', 'caption_space')


def caption_pool(cfg):
    """Teacher-space caption vectors in the order of the API caption pool (caption_space omni), else None.

    A model in the default InternVideo2 caption space uses uniav_api/assets/caption_pool.npz as is.
    """
    space = cfg['dataset'].get('caption_space', 'onepeace')
    assert space != 'onepeace', 'ONE-PEACE caption-space models are no longer supported by uniav_api'
    if space == 'iv2':
        return None
    texts = [str(t) for t in np.load(os.path.join(ROOT, 'uniav_api', 'assets', 'caption_pool.npz'))['texts']]
    z = np.load(os.path.join(ROOT, cfg['dataset']['omni_emb_file']))
    e = np.load(os.path.join(ROOT, 'data', 'youcookii', 'caption_emb_iv2.npz'), allow_pickle=True)
    key = {}
    for k, t in zip(e['keys'], e['sentences']):
        if str(k) + '__text' in z.files:
            key.setdefault(str(t), str(k) + '__text')
    return {'emb': np.stack([z[key[t]] for t in texts]).astype(np.float16)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run'); p.add_argument('--out', default='')
    a = p.parse_args()
    rec = json.load(open(glob.glob(os.path.join(ROOT, 'experiments', 'runs', '*-%s.json' % a.run))[0]))
    cfg = rec['config']
    ck = torch.load(os.path.join(ROOT, rec['out_dir'], 'best_cap.pth.tar'), map_location='cpu', weights_only=False)
    sd = {k: v.half() if v.is_floating_point() else v for k, v in upgrade_state_dict(ck['state_dict']).items()
          if not k.startswith('omni_')}
    dataset = {k: cfg['dataset'][k] for k in DATASET_KEYS if k in cfg['dataset']}
    dataset.setdefault('feat_source', 'onepeace')
    out = a.out or os.path.join(ROOT, 'ckpt', 'api', 'uniav_%s.pth' % a.run)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'state_dict': sd, 'config': {'dataset': dataset, 'model': cfg['model']},
                'caption_pool': caption_pool(cfg),
                'run': a.run, 'commit': rec['git']['commit'], 'epoch': ck['epoch'] + 1,
                'final_eval': rec.get('final_eval')}, out)
    n = sum(v.numel() for v in sd.values())
    print('%s: %.1fM parameters, %.0f MB, features %s, caption space %s, epoch %d'
          % (out, n / 1e6, os.path.getsize(out) / 1e6, dataset['feat_source'],
             dataset.get('caption_space'), ck['epoch'] + 1))


if __name__ == '__main__':
    main()
