"""Pack a training run into one self-describing checkpoint for uniav_api.

  python tools/export_api_ckpt.py iv2                # -> ckpt/api/uniav_iv2.pth

Keeps the best captioning checkpoint (best_cap) in fp16, without the OmniRetriever projection
(training only), plus the run's dataset / model config and its final evaluation, so the API
knows which features the model expects without configs/youcook2_event.yaml.
"""
import argparse
import glob
import json
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_KEYS = ('feat_source', 'feat_stride', 'num_frames', 'default_fps', 'max_seq_len', 'force_upsampling',
                'iv2_video_keys', 'iv2_l2norm')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run'); p.add_argument('--out', default='')
    a = p.parse_args()
    rec = json.load(open(glob.glob(os.path.join(ROOT, 'experiments', 'runs', '*-%s.json' % a.run))[0]))
    cfg = rec['config']
    ck = torch.load(os.path.join(ROOT, rec['out_dir'], 'best_cap.pth.tar'), map_location='cpu', weights_only=False)
    sd = {k: v.half() if v.is_floating_point() else v for k, v in ck['state_dict'].items()
          if not k.startswith('omni_')}
    dataset = {k: cfg['dataset'][k] for k in DATASET_KEYS if k in cfg['dataset']}
    dataset.setdefault('feat_source', 'onepeace')
    out = a.out or os.path.join(ROOT, 'ckpt', 'api', 'uniav_%s.pth' % a.run)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'state_dict': sd, 'config': {'dataset': dataset, 'model': cfg['model']},
                'run': a.run, 'commit': rec['git']['commit'], 'epoch': ck['epoch'] + 1,
                'final_eval': rec.get('final_eval')}, out)
    n = sum(v.numel() for v in sd.values())
    print('%s: %.1fM parameters, %.0f MB, features %s, epoch %d'
          % (out, n / 1e6, os.path.getsize(out) / 1e6, dataset['feat_source'], ck['epoch'] + 1))


if __name__ == '__main__':
    main()
