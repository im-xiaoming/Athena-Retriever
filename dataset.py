"""YouCook2 for event segmentation + captioning.

One item = one video:
  - its features (InternVideo2 video v768 + BEATs audio a768, one row per second) resampled to 256 steps
  - its GT steps on that 256-step grid, and each step's label = its caption index inside the video (0..n-1)
  - the training targets of every pyramid point (gt_cls, gt_off)
  - the caption vectors of its steps
"""
import json
import os
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import config as C


def make_points():
    """Pyramid points, one row (t, regression range lo, hi, stride) per point; 504 points for 256 steps.
    t is the position on the level-0 grid. A level only learns segments whose larger boundary distance
    is inside its regression range, so short steps go to fine levels and long ones to coarse levels."""
    points = []
    for stride, (lo, hi) in zip(C.FPN_STRIDES, C.REGRESSION_RANGE):
        t = torch.arange(0, C.MAX_SEQ_LEN, stride).float()[:, None]
        rest = torch.tensor([lo, hi, stride]).float()[None].repeat(len(t), 1)
        points.append(torch.cat((t, rest), dim=1))
    return torch.cat(points)   # (504, 4)


def label_points(points, segments, labels):
    """Training targets of every point (from ActionFormer). A point is positive for a GT step when it lies
    inside it and the larger of its two boundary distances is in the level's regression range; when several
    steps qualify the shortest wins.
      gt_cls (P, NMAX) one-hot caption index of the point's step (all zeros = background)
      gt_off (P, 2)    distances to that step's start and end, in units of the level's stride
    """
    P = len(points)
    left = points[:, 0, None] - segments[None, :, 0]
    right = segments[None, :, 1] - points[:, 0, None]
    reg = torch.stack((left, right), dim=-1)                      # (P, N, 2)
    inside = reg.min(-1)[0] > 0
    max_dist = reg.max(-1)[0]
    in_range = (max_dist >= points[:, 1, None]) & (max_dist <= points[:, 2, None])

    lens = (segments[:, 1] - segments[:, 0])[None].repeat(P, 1)  # (P, N)
    lens[~inside] = float('inf')
    lens[~in_range] = float('inf')
    min_len, min_idx = lens.min(dim=1)
    chosen = (lens <= min_len[:, None] + 1e-3) & (lens < float('inf'))

    gt_cls = (chosen.float() @ F.one_hot(labels, C.NMAX).float()).clamp(0, 1)
    gt_off = reg[torch.arange(P), min_idx] / points[:, 3, None]
    return gt_cls, gt_off


def random_crop(fv, fa, segments, labels):
    """Training augmentation: keep a random window of 90-100% of the steps. A GT step survives when at
    least half of it is inside the window."""
    T = fv.shape[1]
    new_len = random.randint(round(C.CROP_RATIO[0] * T), round(C.CROP_RATIO[1] * T))
    if new_len == T:
        return fv, fa, segments, labels
    for _ in range(200):   # try windows until one keeps at least one step
        st = random.randint(0, T - new_len)
        ed = st + new_len
        left = torch.clamp(segments[:, 0], min=st - 0.5)
        right = torch.clamp(segments[:, 1], max=ed + 0.5)
        inter = (right - left).clamp(min=0)
        keep = inter / (segments[:, 1] - segments[:, 0]) >= C.TRUNC_THRESH
        if keep.any():
            break
    segments = torch.stack((left[keep], right[keep]), dim=1) - st
    return fv[:, st:ed], fa[:, st:ed], segments, labels[keep]


# caption vectors and texts of every video, keyed "<video_id>#<caption index>"
_z = np.load(C.CAPTION_FILE, allow_pickle=True)
CAP_EMB = {k: v for k, v in zip(_z['keys'], _z['emb'])}
CAP_TEXT = {k: str(s) for k, s in zip(_z['keys'], _z['sentences'])}
POINTS = make_points()


def load_videos(subset):
    """Videos of a subset ('training' / 'validation') that have caption vectors and features:
    their id, duration and GT steps in seconds (at most NMAX)."""
    with open(C.ANNO_FILE) as f:
        db = json.load(f)['database']
    videos = []
    for vid, v in db.items():
        anns = v['annotations'][:C.NMAX]
        if v['subset'] != subset or len(anns) == 0:
            continue
        if any('%s#%d' % (vid, i) not in CAP_EMB for i in range(len(anns))):
            continue
        if not os.path.exists('%s/%s.npz' % (C.FEAT_FOLDER, vid)):
            continue
        videos.append({'id': vid, 'duration': v['duration'], 'n_cap': len(anns),
                       'segments': np.array([a['segment'] for a in anns], dtype=np.float32)})
    return videos


def build_pool(videos):
    """Unique train captions (identical sentences share one index). They are the negatives of the
    embedding loss and the captions picked from at inference.
    Returns pool_text [N], pool_emb (N, 512) and cap_pool_idx {"<video_id>#<i>": index in the pool}."""
    pool_text, pool_emb, index, cap_pool_idx = [], [], {}, {}
    for v in videos:
        for i in range(v['n_cap']):
            k = '%s#%d' % (v['id'], i)
            t = CAP_TEXT[k]
            if t not in index:
                index[t] = len(pool_text)
                pool_text.append(t)
                pool_emb.append(CAP_EMB[k])
            cap_pool_idx[k] = index[t]
    return pool_text, np.stack(pool_emb).astype(np.float32), cap_pool_idx


def make_item(v, is_training, cap_pool_idx=None):
    """Everything the model needs for one video (see the module docstring)."""
    vid = v['id']

    # features: L2-normalise every row, then resample the n rows (one per second) to 256 steps
    z = np.load('%s/%s.npz' % (C.FEAT_FOLDER, vid))
    fv = z['v768'].astype(np.float32)
    fa = z['a768'].astype(np.float32)
    fv = fv / np.linalg.norm(fv, axis=1, keepdims=True)
    fa = fa / np.linalg.norm(fa, axis=1, keepdims=True)
    n = min(len(fv), len(fa))
    fv = torch.from_numpy(fv[:n].T.copy())[None]   # (1, 768, n)
    fa = torch.from_numpy(fa[:n].T.copy())[None]
    fv = F.interpolate(fv, size=C.MAX_SEQ_LEN, mode='linear', align_corners=False)[0]   # (768, 256)
    fa = F.interpolate(fa, size=C.MAX_SEQ_LEN, mode='linear', align_corners=False)[0]

    # time grid: one step = n / 256 seconds, step i is centred on (i + 0.5) * step_sec,
    # so a time t (seconds) is at grid position t / step_sec - 0.5
    step_sec = n / C.MAX_SEQ_LEN
    segments = torch.from_numpy(v['segments'] / step_sec - 0.5)
    labels = torch.arange(v['n_cap'])

    if is_training:
        # drop GT steps that start after the features end; keep a step cut by the end only if at
        # least half of it remains (at least one step is always kept)
        end = C.MAX_SEQ_LEN + 0.5
        keep = []
        for i, (s, e) in enumerate(segments.tolist()):
            if s < end and e > s and (min(e, end) - s) / (e - s) >= C.TRUNC_THRESH:
                keep.append(i)
        if not keep:
            keep = [0]
        segments = segments[keep].clamp(max=end)
        labels = labels[keep]
        fv, fa, segments, labels = random_crop(fv, fa, segments, labels)

    gt_cls, gt_off = label_points(POINTS, segments, labels)

    # caption vectors of the video, padded to NMAX
    cap_emb = torch.zeros(C.NMAX, C.CLIP_DIM)
    pool_idx = torch.full((C.NMAX,), -1)
    texts = []
    for i in range(v['n_cap']):
        k = '%s#%d' % (vid, i)
        cap_emb[i] = torch.from_numpy(CAP_EMB[k])
        texts.append(CAP_TEXT[k])
        if is_training:
            pool_idx[i] = cap_pool_idx[k]

    return {'video_id': vid, 'visual': fv, 'audio': fa, 'segments': segments, 'labels': labels,
            'gt_cls': gt_cls, 'gt_off': gt_off, 'cap_emb': cap_emb, 'cap_pool_idx': pool_idx,
            'cap_text': texts, 'step_sec': step_sec,
            'duration': v['duration'], 'segments_sec': v['segments'].tolist()}


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch


def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)


class YouCook2Dataset(Dataset):
    """One item per video (make_item). The training set also builds the caption pool."""

    def __init__(self, subset, is_training):
        self.is_training = is_training
        self.videos = load_videos(subset)
        self.cap_pool_idx = None
        if is_training:
            self.pool_text, self.pool_emb, self.cap_pool_idx = build_pool(self.videos)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return make_item(self.videos[idx], self.is_training, self.cap_pool_idx)
