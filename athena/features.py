"""Input features of the event model: which ones, their time grid, and how they become model input.

InternVideo2 vision (v768 and/or v512, the checkpoint's iv2_video_keys) + BEATs audio (a768), one
row per second. Stored as <video_id>.npz (keys v768, v512, a768), e.g. athena/samples/ or
data/youcookii/iv2_feats/. Rows are L2-normalised when training did (iv2_l2norm), and resampled
to max_seq_len steps exactly as libs/datasets/youcook2_cap.py does.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F


def _l2(x):
    """L2-normalise every row of x (T, C)."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


class FeatureSpec:
    """The input format a checkpoint was trained with, read from its dataset config: which video
    keys, row rate, normalisation and grid length."""

    def __init__(self, dataset_cfg):
        d = dataset_cfg
        self.source = d.get('feat_source')
        if self.source != 'iv2':
            raise ValueError('athena takes InternVideo2-feature checkpoints only (this one: feat_source %r)'
                             % self.source)
        self.max_seq_len = d['max_seq_len']
        self.fps = d['default_fps']
        self.video_keys = list(d.get('iv2_video_keys', ['v768']))
        self.l2norm = d.get('iv2_l2norm', True)
        self.stride = self.fps // d.get('iv2_rows_per_sec', 1)   # frames between rows; each row covers 1 s
        self.window = self.fps

    # ------------------------------------------------------------------ loading
    def from_npz(self, z):
        """InternVideo2 arrays (a loaded .npz or a dict with v768 / v512 / a768) -> (visual, audio)."""
        visual = np.concatenate([np.asarray(z[k], np.float32) for k in self.video_keys], axis=1)
        return visual, np.asarray(z['a768'], np.float32)

    def from_store(self, video_id, folder):
        """Features of one video from a folder of stored features, in this checkpoint's format."""
        return self.from_npz(np.load(os.path.join(folder, video_id + '.npz')))

    def stored_ids(self, folder):
        """Ids of the videos with a <id>.npz in folder."""
        return sorted(f[:-4] for f in os.listdir(folder) if f.endswith('.npz'))

    # ------------------------------------------------------------------ model input and time grid
    def prepare(self, visual, audio):
        """(T, C) arrays -> two (C, max_seq_len) float tensors, plus the number of rows used."""
        n = min(len(visual), len(audio))
        visual, audio = np.asarray(visual[:n], np.float32), np.asarray(audio[:n], np.float32)
        if self.l2norm:   # idempotent, so already-normalised input is fine
            visual = np.concatenate([_l2(p) for p in self._split(visual)], axis=1)
            audio = _l2(audio)
        fv = torch.from_numpy(np.ascontiguousarray(visual.T))
        fa = torch.from_numpy(np.ascontiguousarray(audio.T))
        if n != self.max_seq_len:
            fv = F.interpolate(fv[None], size=self.max_seq_len, mode='linear', align_corners=False)[0]
            fa = F.interpolate(fa[None], size=self.max_seq_len, mode='linear', align_corners=False)[0]
        return fv, fa, n

    def _split(self, visual):
        """Split concatenated v768 / v512 columns so each part is normalised on its own, as in training."""
        out, c = [], 0
        for k in self.video_keys:
            w = 512 if k == 'v512' else 768
            out.append(visual[:, c:c + w]); c += w
        return out

    def to_seconds(self, segs, n, duration):
        """Segments in grid units (of max_seq_len steps) -> seconds, as the dataset defines the grid
        (libs/datasets/youcook2_cap.py, __getitem__): n rows resampled to max_seq_len steps of `step`
        frames each, step i centred on frame (i + 0.5) * step."""
        step = float((n - 1) * self.stride + self.window) / self.max_seq_len   # frames per grid step
        return np.clip((segs * step + 0.5 * step) / self.fps, 0.0, float(duration))

    @property
    def dims(self):
        """(visual, audio) channel counts of the model input."""
        return sum(512 if k == 'v512' else 768 for k in self.video_keys), 768
