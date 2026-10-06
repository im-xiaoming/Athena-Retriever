"""Pack a training run into one self-describing checkpoint for uniav_api.

  python tools/export_api_ckpt.py iv2                # -> ckpt/api/uniav_iv2.pth

Keeps the best captioning checkpoint (best_cap) in fp16, without the OmniRetriever projection
(training only), and the best segmentation checkpoint (best_seg) as state_dict_seg when it is another
epoch: segmentation peaks early and captions late (run ov_R0: R@0.5 55.0 at epoch 7, 52.9 at epoch 12
where CIDEr peaks), so the API segments and grounds with best_seg and captions with best_cap, plus the run's dataset / model config and its final evaluation, so the API
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
    cap_file = cfg['dataset'].get('caption_emb_file', '')
    if space == 'iv2' and os.path.basename(cap_file) != 'caption_emb_iv2.npz':
        # InternVideo2 text centred on another mean (caption_emb_iv2j.npz: joint YouCook2 + COIN): the
        # API's captions and new sentences must use the same vectors and the same mean
        texts = [str(t) for t in np.load(os.path.join(ROOT, 'uniav_api', 'assets', 'caption_pool.npz'))['texts']]
        e = np.load(os.path.join(ROOT, cap_file), allow_pickle=True)
        vec = {}
        for t, v in zip(e['sentences'], e['emb']):
            vec.setdefault(str(t), v)
        return {'emb': np.stack([vec[t] for t in texts]).astype(np.float16), 'mean': e['mean'].astype(np.float32),
                'file': os.path.basename(cap_file)}
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
    p.add_argument('--no-seg', action='store_true', help='only best_cap (one set of weights for everything)')
    a = p.parse_args()
    rec = json.load(open(glob.glob(os.path.join(ROOT, 'experiments', 'runs', '*-%s.json' % a.run))[0]))
    cfg = rec['config']
    ck = torch.load(os.path.join(ROOT, rec['out_dir'], 'best_cap.pth.tar'), map_location='cpu', weights_only=False)
    half = lambda c: {k: v.half() if v.is_floating_point() else v for k, v in upgrade_state_dict(c['state_dict']).items()
                      if not k.startswith('omni_')}
    sd = half(ck)
    extra = {}
    seg_path = os.path.join(ROOT, rec['out_dir'], 'best_seg.pth.tar')
    if not a.no_seg and os.path.exists(seg_path):
        cs = torch.load(seg_path, map_location='cpu', weights_only=False)
        if cs['epoch'] != ck['epoch']:
            extra = {'state_dict_seg': half(cs), 'epoch_seg': cs['epoch'] + 1}
    dataset = {k: cfg['dataset'][k] for k in DATASET_KEYS if k in cfg['dataset']}
    dataset.setdefault('feat_source', 'onepeace')
    out = a.out or os.path.join(ROOT, 'ckpt', 'api', 'uniav_%s.pth' % a.run)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'state_dict': sd, 'config': {'dataset': dataset, 'model': cfg['model']},
                'caption_pool': caption_pool(cfg),
                'run': a.run, 'commit': rec['git']['commit'], 'epoch': ck['epoch'] + 1,
                'final_eval': rec.get('final_eval'), **extra}, out)
    n = sum(v.numel() for v in sd.values())
    print('%s: %.1fM parameters, %.0f MB, features %s, caption space %s, epoch %d%s'
          % (out, n / 1e6, os.path.getsize(out) / 1e6, dataset['feat_source'],
             dataset.get('caption_space'), ck['epoch'] + 1,
             ', segmentation weights of epoch %d' % extra['epoch_seg'] if extra else ''))


if __name__ == '__main__':
    main()
