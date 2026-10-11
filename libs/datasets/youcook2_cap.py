"""YouCook2 dataset for event segmentation + captioning.

  - each segment's label is the CAPTION INDEX within the video (0..N-1), not a global class id
  - every sample carries its captions' text vectors (NMAX, 512) and a validity mask (NMAX,)
  - num_classes = NMAX, the maximum number of captions in one video
so label_points produces gt_cls_labels (P, NMAX) unchanged.

Input features: InternVideo2 video (iv2_video_keys, v768 and/or v512) + BEATs audio a768, one row
per second (iv2_rows_per_sec=2 with tools/make_iv2_dense.py), from tools/extract_internvideo2.py,
resampled to max_seq_len steps.

Captions: InternVideo2 text vectors (caption_emb_file, 512-d, tools/embed_captions.py), in the
space of the v512 video projection. They are the space of the txt_sim metric and of the consensus
pick, and by default (caption_space iv2) also the space the model is trained and picks captions in.
caption_space omni trains in the OmniRetriever-7B text space instead (teacher file, omni_emb_file).
"""
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .data_utils import truncate_feats, label_points
from .loc_generators import PointGenerator

NMAX = 16   # maximum number of captions in one YouCook2 video
# config keys of the removed ONE-PEACE feature path; accepted and ignored so old run configs still load
LEGACY_KEYS = {'feat_folder', 'feat_stride', 'num_frames', 'downsample_rate', 'class_aware', 'file_prefix',
               'file_ext', 'multi_modal', 'require_iv2', 'caption_emb_iv2_file'}


class YouCook2CaptionDataset(Dataset):
    """One item per video: its features resampled to max_seq_len steps, its GT steps in grid units,
    the training targets of every pyramid point and the caption vectors of its steps (see __getitem__).

    Other sources in the same file formats (ActivityNet Captions, HT-Step: tools/extra_data.py) use it
    as it is; COIN (CoinCaptionDataset) subclasses it and only replaces _load_captions and _load_json_db.
    pad_omni: a training source without teacher vectors next to one that has them; its items carry
    cap_av_idx -1 so a mixed batch has the key everywhere.

    Feature augmentation, training items only (all off by default):
      aug_stretch     the crop (crop_ratio) is resampled back to max_seq_len steps, so its events get longer
      aug_modal_drop  probability that one stream, audio or visual (even odds), is zeroed
      aug_mask_spans  number of random spans zeroed in each stream, each 1..aug_mask_width steps long
      aug_noise       std of the Gaussian noise added to the features (rows are L2-normalised)
    """

    def __init__(
        self, is_training, split, json_file, caption_emb_file, default_fps, max_seq_len,
        max_buffer_len_factor, scale_factor, regression_range, backbone_arch, trunc_thresh,
        crop_ratio, num_classes, force_upsampling=True, omni_emb_file=None, feat_source='iv2',
        iv2_folder='./data/youcookii/iv2_feats', iv2_video_keys=('v768',), iv2_l2norm=True,
        iv2_rows_per_sec=1, caption_space='iv2', center_sample_radius=0.0, pad_omni=False,
        aug_stretch=False, aug_modal_drop=0.0, aug_mask_spans=0, aug_mask_width=4, aug_noise=0.0, **legacy,
    ):
        unknown = set(legacy) - LEGACY_KEYS
        assert not unknown, 'unknown dataset options %s' % sorted(unknown)
        assert feat_source == 'iv2', 'only InternVideo2 features are supported (feat_source %s)' % feat_source
        assert os.path.isdir(iv2_folder), iv2_folder
        assert os.path.exists(json_file) and os.path.exists(caption_emb_file)
        assert num_classes == NMAX, 'num_classes must be %d' % NMAX
        self.iv2_folder = iv2_folder
        self.iv2_video_keys = list(iv2_video_keys)
        self.iv2_l2norm = iv2_l2norm
        self.iv2_rows_per_sec = iv2_rows_per_sec   # 2: tools/make_iv2_dense.py (iv2_folder: .../iv2_dense)
        self.force_upsampling = force_upsampling
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.crop_ratio = crop_ratio
        self.center_radius = center_sample_radius   # ActionFormer center sampling (training targets only)
        self.aug_stretch, self.aug_modal_drop, self.aug_noise = aug_stretch, aug_modal_drop, aug_noise
        self.aug_mask_spans, self.aug_mask_width = aug_mask_spans, aug_mask_width

        self.source = 0          # dataset id: rows of the pool loss only see captions of their own dataset
        self.pool_offset = 0     # where this dataset's captions start in a pool shared by several datasets
        self._load_captions(caption_emb_file)

        self.data_list = self._load_json_db(json_file)
        assert self.data_list, 'no %s video of %s has both features (%s) and caption vectors' % (split, json_file, iv2_folder)
        self._build_pool()
        self.omni = bool(omni_emb_file) and is_training   # only the train split needs the teacher
        self.pad_omni = pad_omni and not self.omni
        if self.omni:
            self._load_omni(omni_emb_file)
        assert caption_space in ('iv2', 'omni'), 'caption_space %s (ONE-PEACE was removed)' % caption_space
        assert self.emb_dim == 512, '%s is not the InternVideo2 caption file' % caption_emb_file
        self.caption_space = caption_space
        if caption_space == 'omni' and is_training:
            assert self.omni and self.omni_text_ok.all(), 'caption_space omni needs teacher text for every caption'
            self.pool_train = self.omni_text_pool.astype(np.float32)
        else:
            self.pool_train = self.pool_emb

        self.fpn_strides = [scale_factor ** i for i in range(backbone_arch[-1] + 1)]
        for s in self.fpn_strides:
            assert max_seq_len % s == 0, 'max_seq_len must be divisible by every fpn stride'
        self.point_generator = PointGenerator(
            max_seq_len_ori=self.max_seq_len, max_buffer_len_factor=max_buffer_len_factor,
            fpn_levels=len(self.fpn_strides), scale_factor=scale_factor,
            regression_range=regression_range, max_div_factor=max(self.fpn_strides))

    def _load_captions(self, path):
        """InternVideo2 caption vectors keyed "<video_id>#<segment index>", plus the "other" text if stored."""
        z = np.load(path, allow_pickle=True)
        self.cap_emb = {k: v for k, v in zip(z['keys'], z['emb'])}
        self.cap_text = {k: s for k, s in zip(z['keys'], z['sentences'])}
        self.emb_dim = z['emb'].shape[1]
        self.other = z['other'].astype(np.float32) if 'other' in z.files else None

    def _build_pool(self):
        """Unique caption pool of this split; identical captions share one index.

        Only the train pool is used: as negatives during training and as the pool
        captions are picked from at inference.
        """
        self.pool_text, self.pool_index, self.pool_keys, emb = [], {}, [], []
        self.cap_pool_idx = {}
        for it in self.data_list:
            for i in range(it['n_cap']):
                k = '%s#%d' % (it['id'], i)
                t = str(self.cap_text[k])
                if t not in self.pool_index:   # first time this sentence is seen: new pool entry
                    self.pool_index[t] = len(self.pool_text)
                    self.pool_text.append(t)
                    self.pool_keys.append(k)
                    emb.append(self.cap_emb[k])
                self.cap_pool_idx[k] = self.pool_index[t]
        self.pool_emb = np.stack(emb).astype(np.float32)

    def _load_omni(self, path):
        """OmniRetriever-7B vectors of every event clip, keys "<video>#<i>__av" and "__text".

        Builds two pools aligned with the caption pool:
          omni_text_pool (N, 3584) : caption vectors, same order as pool_text
          omni_av_pool   (K, 3584) : audio+video vector of every GT clip, the teacher
        Clips that failed extraction are marked invalid and left out of the loss.
        """
        z = np.load(path)
        have = set(z.files)
        self.omni_dim = z[z.files[0]].shape[0]
        txt = np.zeros((len(self.pool_text), self.omni_dim), dtype=np.float16)
        self.omni_text_ok = np.zeros(len(self.pool_text), dtype=bool)
        av, self.cap_av_idx, self.av_text_idx = [], {}, []
        for it in self.data_list:
            for i in range(it['n_cap']):
                k = '%s#%d' % (it['id'], i)
                j = self.cap_pool_idx[k]
                if not self.omni_text_ok[j] and k + '__text' in have:
                    txt[j] = z[k + '__text']; self.omni_text_ok[j] = True
                if k + '__av' in have:
                    self.cap_av_idx[k] = len(av); av.append(z[k + '__av'])
                    self.av_text_idx.append(j)   # caption of this clip, for a fused T+V+A target
        self.omni_text_pool = txt
        self.omni_av_pool = (np.stack(av) if av else
                             np.zeros((0, self.omni_dim), dtype=np.float16))
        print('Omni teacher : %d/%d caption vectors, %d clip vectors from %s' % (
            self.omni_text_ok.sum(), len(self.pool_text), len(av), path), flush=True)

    def _load_json_db(self, json_file):
        """Videos of this split that have caption vectors and features: their GT steps (at most NMAX)
        in seconds, and labels 0..n-1 (each step's caption index within the video)."""
        with open(json_file) as f:
            db = json.load(f)['database']
        out = []
        for vid, value in db.items():
            if value['subset'].lower() not in self.split:
                continue
            anns = value['annotations'][:NMAX]
            if len(anns) == 0:
                continue
            if any('%s#%d' % (vid, i) not in self.cap_emb for i in range(len(anns))):
                continue
            if not os.path.exists(os.path.join(self.iv2_folder, vid + '.npz')):
                continue   # not extracted (yet)
            segs = np.array([a['segment'] for a in anns], dtype=np.float32)
            out.append({
                'id': vid,
                'fps': self.default_fps if self.default_fps is not None else value['fps'],
                'duration': value['duration'],
                'segments': segs,
                'labels': np.arange(len(anns), dtype=np.int64),   # caption index
                'n_cap': len(anns),
            })
        return tuple(out)

    def __len__(self):
        return len(self.data_list)

    @property
    def feat_dims(self):
        """(visual, audio) channel counts of the model input."""
        return sum(512 if k == 'v512' else 768 for k in self.iv2_video_keys), 768

    def _load_features(self, vid):
        """Visual and audio rows (n, C), plus their step and window in frames.

        Row i is centred on i / rows_per_sec + 0.5 s: stride 1 / rows_per_sec s, window 1 s.
        """
        z = np.load(os.path.join(self.iv2_folder, vid + '.npz'))
        parts = [z[k].astype(np.float32) for k in self.iv2_video_keys]
        fa = z['a768'].astype(np.float32)
        if self.iv2_l2norm:   # raw norms differ widely: v768 ~54, a768 ~6, v512 is already 1
            parts = [p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-6) for p in parts]
            fa = fa / np.maximum(np.linalg.norm(fa, axis=1, keepdims=True), 1e-6)
        fv = np.concatenate(parts, axis=1)
        n = min(fv.shape[0], fa.shape[0])
        return fv[:n], fa[:n], self.default_fps // self.iv2_rows_per_sec, self.default_fps

    def _augment(self, d, feat_offset):
        """Feature augmentation of a training item after the crop, in place (see the class docstring)."""
        T = d['feats']['visual'].shape[1]
        if self.aug_stretch and T < self.max_seq_len:
            # grid position p of a time becomes (p + offset) * scale - offset, as linear resampling maps step centres
            scale = self.max_seq_len / T
            for k in ('visual', 'audio'):
                d['feats'][k] = F.interpolate(d['feats'][k][None], size=self.max_seq_len, mode='linear',
                                              align_corners=False)[0]
            d['segments'] = (d['segments'] + feat_offset) * scale - feat_offset
            d['feat_stride'] /= scale
            d['feat_num_frames'] /= scale
            T = self.max_seq_len
        for k in ('visual', 'audio'):
            f = d['feats'][k].clone()
            for _ in range(self.aug_mask_spans):
                w = random.randint(1, min(self.aug_mask_width, T))
                st = random.randint(0, T - w)
                f[:, st:st + w] = 0
            if self.aug_noise > 0:
                f = f + self.aug_noise * torch.randn_like(f)
            d['feats'][k] = f
        if self.aug_modal_drop > 0 and random.random() < self.aug_modal_drop:
            k = random.choice(('visual', 'audio'))
            d['feats'][k] = torch.zeros_like(d['feats'][k])

    def __getitem__(self, idx):
        """Returns a dict:
          video_id, fps, duration, n_cap
          feats         {'visual': (C_v, T), 'audio': (C_a, T)}, T = max_seq_len
          segments      (N, 2) GT steps in grid units; labels (N,) their caption index in the video
          feat_stride   frames per grid step (with feat_num_frames: grid units <-> seconds)
          points        pyramid points, list of (T_l, 4): (t, regression range lo, hi, stride)
          gt_cls_labels (P, NMAX) one-hot caption index of the step each point belongs to (zeros: none)
          gt_offsets    (P, 2) distances from each point to its step's start and end, in strides
          cap_emb (NMAX, 512), cap_mask (NMAX,), cap_text [NMAX]  caption vectors / texts of the steps
          cap_pool_idx  (NMAX,) index of each caption in the shared train caption pool (-1: padding)
          cap_av_idx    (NMAX,) index of each step's teacher clip vector (-1: none), with the teacher
          source        dataset id; segments_sec  GT steps in seconds, for scoring
        """
        item = self.data_list[idx]
        vid = item['id']
        fv, fa, stride, window = self._load_features(vid)
        # Time grid. The n feature rows are resampled to max_seq_len steps, so one step spans
        # feat_stride frames (at default_fps) and step i is centred on frame (i + 0.5) * feat_stride.
        # A time t in seconds is therefore at grid position t * fps / feat_stride - 0.5
        # (feat_offset = 0.5); the model converts back with (pos * feat_stride + 0.5 * num_frames) / fps.
        feat_stride = float((fv.shape[0] - 1) * stride + window) / self.max_seq_len
        num_frames = feat_stride   # frames seen by one step: exactly one step's worth
        feat_offset = 0.5 * num_frames / feat_stride
        fv = torch.from_numpy(np.ascontiguousarray(fv.transpose()))
        fa = torch.from_numpy(np.ascontiguousarray(fa.transpose()))
        if fv.shape[-1] != self.max_seq_len and self.force_upsampling:
            fv = F.interpolate(fv[None], size=self.max_seq_len, mode='linear', align_corners=False)[0]
            fa = F.interpolate(fa[None], size=self.max_seq_len, mode='linear', align_corners=False)[0]
        feats = {'visual': fv, 'audio': fa}

        segments = torch.from_numpy(item['segments'] * item['fps'] / feat_stride - feat_offset)
        labels = torch.from_numpy(item['labels'])
        if self.is_training:
            # drop GT steps that start after the features end; a step cut by the end is kept (clamped)
            # only if at least trunc_thresh of it remains; at least one step is always kept
            vid_len = feats['visual'].shape[1] + feat_offset
            keep_s, keep_l = [], []
            for seg, lab in zip(segments, labels):
                if seg[0] >= vid_len:
                    continue
                denom = seg[1].item() - seg[0].item()
                if denom <= 0:
                    continue
                if (min(seg[1].item(), vid_len) - seg[0].item()) / denom >= self.trunc_thresh:
                    keep_s.append(seg.clamp(max=vid_len)); keep_l.append(lab.view(1))
            if len(keep_s) == 0:
                keep_s, keep_l = [segments[0].clamp(max=vid_len)], [labels[0].view(1)]
            segments = torch.stack(keep_s, dim=0); labels = torch.cat(keep_l)

        data_dict = {
            'video_id': vid, 'feats': feats, 'segments': segments, 'labels': labels,
            'fps': item['fps'], 'duration': item['duration'],
            'feat_stride': feat_stride, 'feat_num_frames': num_frames,
            'n_cap': item['n_cap'],
        }
        if self.is_training:
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio)
            self._augment(data_dict, feat_offset)

        points = self.point_generator(self.fpn_strides, data_dict['feats']['visual'], self.is_training)
        data_dict['gt_cls_labels'], data_dict['gt_offsets'] = label_points(
            points, data_dict['segments'], data_dict['labels'], self.num_classes, False,
            self.center_radius if self.is_training else 0.0)
        data_dict['points'] = points

        # caption matrix of the video, padded to NMAX
        emb = np.zeros((NMAX, self.emb_dim), dtype=np.float32)
        cmask = np.zeros((NMAX,), dtype=bool)
        texts = [''] * NMAX
        pidx = np.full((NMAX,), -1, dtype=np.int64)
        for i in range(item['n_cap']):
            k = '%s#%d' % (vid, i)
            emb[i] = self.cap_emb[k]
            cmask[i] = True
            texts[i] = str(self.cap_text[k])
            pidx[i] = self.cap_pool_idx[k] + self.pool_offset
        data_dict['cap_emb'] = torch.from_numpy(emb)
        data_dict['cap_mask'] = torch.from_numpy(cmask)
        data_dict['cap_pool_idx'] = torch.from_numpy(pidx)
        if self.omni:
            aidx = np.full((NMAX,), -1, dtype=np.int64)
            for i in range(item['n_cap']):
                aidx[i] = self.cap_av_idx.get('%s#%d' % (vid, i), -1)
            data_dict['cap_av_idx'] = torch.from_numpy(aidx)
        elif self.pad_omni:
            data_dict['cap_av_idx'] = torch.full((NMAX,), -1, dtype=torch.int64)
        data_dict['cap_text'] = texts
        data_dict['source'] = self.source
        # original boundaries in seconds, used only for scoring
        data_dict['segments_sec'] = item['segments'].tolist()
        return data_dict
