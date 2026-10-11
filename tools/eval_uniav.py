"""Run the original UniAV checkpoint on the benchmark test sets and save its predictions for tools/eval_benchmark.py.

  python tools/eval_uniav.py [--sets dcase unav100] [--ckpt ckpt/uniav/model_epoch_006.pth.tar]

The upstream code in benchmark/UniAV is not modified; this wrapper does what its eval.py cannot do here:
  - single GPU, no torch.distributed (eval.py calls dist.get_rank() unconditionally)
  - the pyramid points are moved to the GPU with the batch
  - the class prompts are read from benchmark/UniAV/data/<folder>/ (eval.py expects ./data/dcase/ for DCASE)
  - features and annotations are read from benchmark/<folder>/ (tools/benchmark_data.py fetch --uniav)
It must run in its own process: UniAV's `libs` package has the same name as ours.
Also prints UniAV's own class-aware mAP (its ANETdetection), to check the run against the paper.

Writes benchmark/preds/uniav_<set>.json:
  {"model": ..., "set": ..., "videos": [every evaluated id], "results": {id: [[start, end, score, label_id], ...]}}
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIAV = os.path.join(ROOT, 'benchmark', 'UniAV')
sys.path.insert(0, UNIAV)
from libs.core import load_config                       # noqa: E402  (UniAV's libs)
from libs.datasets import make_dataset                  # noqa: E402
from libs.datasets.data_utils import trivial_batch_collator   # noqa: E402
from libs.modeling import make_multimodal_meta_arch    # noqa: E402
from libs.utils import ANETdetection, fix_random_seed  # noqa: E402

# set name -> (UniAV task, folder in benchmark/ and in benchmark/UniAV/data/, prompt file)
SETS = {'dcase': ('TASK3', 'desed', 'dcase_prompt.npy'), 'unav100': ('TASK2', 'unav100', 'unav100_prompt.npy')}
PROMPTS = {'TASK1': ('activitynet13', 'anet_prompt.npy')}   # ActivityNet: no features, but the model needs its prompts


def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fix_random_seed(0, include_cuda=True)
    cfg, task_cfg = load_config(os.path.join(UNIAV, 'configs', 'multi_task_anet_unav_dcase.yaml'))
    prompts = dict(PROMPTS, **{t: (folder, f) for t, folder, f in SETS.values()})
    for task, (folder, f) in prompts.items():
        task_cfg[task]['clip_class_feature'] = torch.tensor(
            np.load(os.path.join(UNIAV, 'data', folder, f)), dtype=torch.float32).to(dev)
    model = make_multimodal_meta_arch(cfg['model_name'], **cfg['model']).to(dev)
    state = torch.load(a.ckpt, map_location='cpu')['state_dict_ema']
    model.load_state_dict({k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()})
    model.eval()
    os.makedirs(a.out, exist_ok=True)

    for name in a.sets:
        task, folder, _ = SETS[name]
        tc = task_cfg[task]
        tc['dataset']['json_file'] = os.path.join(ROOT, 'benchmark', folder, '%s_all.json' % name)
        tc['dataset']['feat_folder'] = os.path.join(ROOT, 'benchmark', folder, 'onepeace_finetune_features_025s')
        ds = make_dataset(tc['dataset_name'], False, tc['test_split'], **tc['dataset'])
        dl = torch.utils.data.DataLoader(ds, batch_size=tc['batch_size'], num_workers=a.workers,
                                         collate_fn=trivial_batch_collator, shuffle=False)
        results, flat = {}, {'video-id': [], 't-start': [], 't-end': [], 'label': [], 'score': []}
        with torch.no_grad():
            for batch in dl:
                for x in batch:   # the dataset builds the pyramid points on the CPU; upstream leaves them there
                    x['points'] = [pt.to(dev) for pt in x['points']]
                for o in model(batch, task, tc['task_type']):
                    segs, sc, lab = o['segments'].numpy(), o['scores'].numpy(), o['labels'].numpy()
                    results[o['video_id']] = [[float(s), float(e), float(c), int(l)]
                                              for (s, e), c, l in zip(segs, sc, lab)]
                    flat['video-id'] += [o['video_id']] * len(segs)
                    for k, v in (('t-start', segs[:, 0]), ('t-end', segs[:, 1]), ('label', lab), ('score', sc)):
                        flat[k].append(v)
        flat = {k: (np.concatenate(v) if k != 'video-id' else v) for k, v in flat.items()}
        print('%s: %d videos, UniAV class-aware mAP (its own evaluator):' % (name, len(results)), flush=True)
        ANETdetection(tc['dataset']['json_file'], tc['test_split'][0],
                      tiou_thresholds=ds.get_attributes()['tiou_thresholds']).evaluate(flat, True, verbose=True)
        out = os.path.join(a.out, 'uniav_%s.json' % name)
        with open(out, 'w') as f:
            json.dump({'model': 'UniAV (%s)' % os.path.relpath(a.ckpt, ROOT), 'set': name,
                       'videos': sorted(results), 'results': results}, f)
        print('-> %s' % out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sets', nargs='+', default=list(SETS), choices=list(SETS))
    p.add_argument('--ckpt', default=os.path.join(ROOT, 'ckpt', 'uniav', 'model_epoch_006.pth.tar'))
    p.add_argument('--out', default=os.path.join(ROOT, 'benchmark', 'preds'))
    p.add_argument('--workers', type=int, default=4)
    main(p.parse_args())
