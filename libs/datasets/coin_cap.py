"""COIN for joint training with YouCook2 and for open-vocabulary evaluation (docs/PLAN_openvocab.md).

Same features, grid and item format as YouCook2CaptionDataset; only where the videos and their
captions come from differs:
  - videos: data/coin/coin_anno.json (tools/coin_data.py anno), filtered by role:
    "train" (seen tasks, COIN training subset), "test_seen" (seen tasks, COIN testing subset),
    "test_unseen" (tasks never used for training)
  - captions: the step labels, as InternVideo2 text vectors centred like the YouCook2 captions
    (data/coin/label_emb_iv2j.npz from tools/embed_captions.py --joint)
Items carry source = 1, so the pool loss compares COIN steps only with COIN labels.
"""
import json
import os

import numpy as np
import torch

from .youcook2_cap import NMAX, YouCook2CaptionDataset


class CoinCaptionDataset(YouCook2CaptionDataset):
    def __init__(self, is_training, roles, anno_file, label_emb_file, iv2_folder, pad_omni=False, **common):
        common = {k: v for k, v in common.items()
                  if k not in ('json_file', 'caption_emb_file', 'iv2_folder', 'omni_emb_file')}
        super().__init__(is_training, list(roles), anno_file, label_emb_file, iv2_folder=iv2_folder,
                         omni_emb_file=None, **common)
        self.source = 1
        self.pad_omni = pad_omni   # the model trains with the YouCook2-only teacher: COIN steps have no clip vector

    def _load_captions(self, path):
        """Every COIN step label with its vector; per-video caption keys are filled in _load_json_db."""
        z = np.load(path)
        self.labels = [str(l) for l in z['labels']]
        self.label_emb = z['emb'].astype(np.float32)
        self.label_vec = dict(zip(self.labels, self.label_emb))
        self.emb_dim = self.label_emb.shape[1]
        self.other = z['other'].astype(np.float32)
        self.cap_emb, self.cap_text = {}, {}

    def _load_json_db(self, anno_file):
        with open(anno_file) as f:
            videos = json.load(f)['videos']
        out = []
        for vid, v in sorted(videos.items()):
            if v['role'] not in self.split or not os.path.exists(os.path.join(self.iv2_folder, vid + '.npz')):
                continue
            segs, labs = v['segments'][:NMAX], v['labels'][:NMAX]   # 47 of 11,827 videos have more than 16 steps
            if not segs:
                continue
            for i, l in enumerate(labs):
                k = '%s#%d' % (vid, i)
                self.cap_emb[k], self.cap_text[k] = self.label_vec[l], l
            out.append({'id': vid, 'fps': self.default_fps, 'duration': v['duration'],
                        'segments': np.array(segs, dtype=np.float32),
                        'labels': np.arange(len(segs), dtype=np.int64), 'n_cap': len(segs),
                        'task': v['task']})
        return tuple(out)

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        if self.pad_omni:
            d['cap_av_idx'] = torch.full((NMAX,), -1, dtype=torch.int64)
        return d
