"""Benchmark test videos (DCASE public_eval, UnAV-100 test) for segmentation only (tools/eval_benchmark.py).

Same features, grid and item format as YouCook2CaptionDataset; the videos and their GT events come from
benchmark/<folder>/<set>_all.json (tools/benchmark_data.py). These sets have class labels, not captions:
each event's caption text is its class name and its caption vector is zero, so nothing caption-related
is scored. Class labels are dropped: events with the same span are one GT event.
The scorer reads the GT from the json itself; the items keep at most NMAX events.
"""
import json
import os

import numpy as np

from .youcook2_cap import NMAX, YouCook2CaptionDataset


class BenchmarkDataset(YouCook2CaptionDataset):
    """split: the json subset ('public_eval' for DCASE, 'test' for UnAV-100); common: the YouCook2 dataset
    options (grid, features), minus the YouCook2 file paths."""

    def __init__(self, split, anno_file, feat_folder, **common):
        common = {k: v for k, v in common.items()
                  if k not in ('json_file', 'caption_emb_file', 'iv2_folder', 'omni_emb_file')}
        # no caption file: the json stands in for it (the parent only checks that the path exists)
        super().__init__(False, [split], anno_file, anno_file, iv2_folder=feat_folder, omni_emb_file=None, **common)

    def _load_captions(self, path):
        self.cap_emb, self.cap_text, self.emb_dim, self.other = {}, {}, 512, None

    def _load_json_db(self, json_file):
        """Videos of the split with features: their distinct GT spans in seconds (first label as the text)."""
        with open(json_file) as f:
            db = json.load(f)['database']
        out = []
        for vid, v in sorted(db.items()):
            if v['subset'].lower() not in self.split or not os.path.exists(os.path.join(self.iv2_folder, vid + '.npz')):
                continue
            spans = {}
            for a in v['annotations']:
                spans.setdefault(tuple(a['segment']), a['label'])
            spans = sorted(spans.items())[:NMAX]
            if not spans:
                continue
            for i, (_, label) in enumerate(spans):
                k = '%s#%d' % (vid, i)
                self.cap_emb[k], self.cap_text[k] = np.zeros(self.emb_dim, dtype=np.float32), label
            out.append({'id': vid, 'fps': self.default_fps, 'duration': v['duration'],
                        'segments': np.array([s for s, _ in spans], dtype=np.float32),
                        'labels': np.arange(len(spans), dtype=np.int64), 'n_cap': len(spans)})
        return tuple(out)
