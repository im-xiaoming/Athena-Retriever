import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset, make_generator
from .data_utils import truncate_feats, label_points

from typing import Literal


@register_dataset("htstep")
class HTStepDataset(Dataset):
    def __init__(
        self,
        is_training,      # True: training split (random crops and the caption pool)
        split: Literal['training', 'validation'],            # list of subsets to keep, e.g. ['training']
        feat_folder,      # feature folder, one <id>.npz per video
        video_key,        # key of the video rows in the npz (k768)
        json_file,        # annotation json
        caption_emb_file, # caption vectors and text
        omni_emb_file,    # OmniRetriever caption vectors; '' = none
        default_fps,      # nominal fps (16), converts seconds <-> steps
        max_seq_len,      # steps per video (256)
        trunc_thresh,     # a cropped segment is kept only if at least this fraction is left (0.5)
        crop_ratio,       # (0.9, 1.0): a random crop keeps 90-100% of the video
        num_classes: int,      # at most this many captions per video (16)
        scale_factor,     # length ratio between two pyramid levels
        regression_range, # regression range of every pyramid level
        backbone_arch,    # (#conv, #stem, #pyramid branch), gives the number of levels
    ):
        self.feat_folder = feat_folder
        self.video_key = video_key
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.num_classes = num_classes

        # caption vectors and text, keys "<video_id>#<segment index>"
        # ['keys', 'emb', 'sentences', 'other', 'mean']
        # emb = (11594, 512)
        z = np.load(caption_emb_file, allow_pickle=True)
        self.cap_emb = {k: v for k, v in zip(z['keys'], z['emb'])}
        self.cap_text = {k: str(s) for k, s in zip(z['keys'], z['sentences'])}
        self.omni_emb = None
        if omni_emb_file:
            t = np.load(omni_emb_file, allow_pickle=True)
            self.omni_emb = {k.replace('__text', ''): np.array(v) for k, v in t.items()}

        # read the annotations, keep the videos of the split
        self.data_list = self._load_json_db(json_file)
        if is_training:
            self._build_pool()

        # point generator of the pyramid
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1]+1)] # [1, 2, 4, 8, 16, 32]
        self.point_generator = make_generator(
            'point',
            **{
                'regression_range': regression_range,
                'max_seq_len_ori': max_seq_len,
                'max_buffer_len_factor': 1.0,
                'fpn_levels': len(self.fpn_strides),
                'scale_factor': scale_factor,
                'max_div_factor': self.fpn_strides[-1],
            }
        )

    def _load_json_db(self, json_file) -> tuple[dict[Literal["id", 'fps', 'duration', 'labels', 'n_cap']]]:
        with open(json_file, 'r') as f:
            json_db = json.load(f)['database']
        dict_db = tuple()
        for key, value in json_db.items():
            if value['subset'].lower() not in self.split: # split = ['traning' or 'validation']
                continue
            anns = value['annotations'][:self.num_classes]
            if len(anns) == 0:
                continue
            # keys: --bv0V6ZjWI#0', '--bv0V6ZjWI#1', '--bv0V6ZjWI#2,...
            if any('%s#%d' % (key, i) not in self.cap_emb for i in range(len(anns))):
                continue
            if self.omni_emb is not None:
                if any('%s#%d' % (key, i) not in self.omni_emb for i in range(len(anns))):
                    continue
            if not os.path.exists(os.path.join(self.feat_folder, key + '.npz')):
                continue   # not extracted (yet)
            segments = np.array([a['segment'] for a in anns], dtype=np.float32) # (N, 2)
            dict_db += ({'id': key,
                         'fps': self.default_fps,
                         'duration': value['duration'], # None
                         'segments': segments, # (n, 2)
                         'labels': np.arange(len(anns), dtype=np.int64),   # caption index; (n,)
                         'n_cap': len(anns), # n
            }, )
        return dict_db

    def _build_pool(self):
        self.pool_text, self.pool_index, emb, omi_emb_ = [], {}, [], []
        self.cap_pool_idx = {}
        for item in self.data_list: # tuple dict_db
            for i in range(item['n_cap']):
                k = '%s#%d' % (item['id'], i)
                t = self.cap_text[k]
                if t not in self.pool_index:   # first time this sentence is seen: new pool entry
                    self.pool_index[t] = len(self.pool_text)
                    self.pool_text.append(t)
                    emb.append(self.cap_emb[k])
                    if self.omni_emb is not None:
                        omi_emb_.append(self.omni_emb[k])
                self.cap_pool_idx[k] = self.pool_index[t]
        self.pool_emb = np.stack(emb).astype(np.float32) # (N, 512)
        self.pool_omni_emb = None
        if self.omni_emb is not None:
            self.pool_omni_emb = np.stack(omi_emb_).astype(np.float32) # (N, 3584)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        vid_id = video_item['id']

        # read the features, L2-normalise every row (a768 is all zeros for videos without audio)
        z = np.load(os.path.join(self.feat_folder, vid_id + '.npz'))
        fv = z[self.video_key].astype(np.float32) # TxC
        fa = z['a768'].astype(np.float32) # TxC
        fv = fv / np.maximum(np.linalg.norm(fv, axis=1, keepdims=True), 1e-6)
        fa = fa / np.maximum(np.linalg.norm(fa, axis=1, keepdims=True), 1e-6)
        t = min(fv.shape[0], fa.shape[0]) # T

        # Time grid. n feature rows (one per second = default_fps frames) are resampled to max_seq_len steps,
        # so one step is feat_stride frames long and step i is centred on frame (i + 0.5) * feat_stride.
        # Time t (seconds) sits at grid position t * fps / feat_stride - 0.5;
        # the model maps back with (position * feat_stride + 0.5 * num_frames) / fps.
        feat_stride = float(t * self.default_fps) / self.max_seq_len # total frame: T * 16
        num_frames = feat_stride
        feat_offset = 0.5
        fv = torch.from_numpy(np.ascontiguousarray(fv[:t].transpose())) # (768, T)
        fa = torch.from_numpy(np.ascontiguousarray(fa[:t].transpose())) # (768, T)
        #                   (1, 768, T)
        fv = F.interpolate(fv[None], size=self.max_seq_len, mode='linear', align_corners=False)[0] # (768, 256)
        fa = F.interpolate(fa[None], size=self.max_seq_len, mode='linear', align_corners=False)[0] # (768, 256)

        segments = torch.from_numpy(video_item['segments'] * video_item['fps'] / feat_stride - feat_offset) # (n, 2)
        labels = torch.from_numpy(video_item['labels']) # (n,)
        if self.is_training:
            # drop GT steps that start after the features end; a step cut at the end is kept only if
            # at least trunc_thresh of it is left; always keep at least one step
            vid_len = self.max_seq_len + feat_offset
            keep = [i for i, (s, e) in enumerate(segments.tolist()) if s < vid_len and e > s and (min(e, vid_len) - s) / (e - s) >= self.trunc_thresh]
            keep = keep or [0]
            segments, labels = segments[keep].clamp(max=vid_len), labels[keep]

        data_dict = {'video_id': vid_id,
                     'feats': {'visual': fv, 'audio': fa},
                     'segments': segments,
                     'labels': labels,
                     'fps': video_item['fps'],
                     'duration': video_item['duration'],
                     'feat_stride': feat_stride,
                     'feat_num_frames': num_frames}

        # random crop of the features during training
        if self.is_training:
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio)

        # training targets of every pyramid point
        points = self.point_generator(self.fpn_strides, data_dict['feats']['visual'], self.is_training)

        data_dict['gt_cls_labels'], data_dict['gt_offsets'] = label_points(
            points, data_dict['segments'], data_dict['labels'], self.num_classes)
        data_dict['points'] = points

        # caption matrix of the video, padded to NMAX (16)
        emb = np.zeros((self.num_classes, 512), dtype=np.float32)
        pidx = np.full((self.num_classes,), -1, dtype=np.int64)
        texts = []
        for i in range(video_item['n_cap']): # n
            k = '%s#%d' % (vid_id, i)
            emb[i] = self.cap_emb[k]
            texts.append(self.cap_text[k])
            if self.is_training:
                pidx[i] = self.cap_pool_idx[k]
        data_dict['cap_emb'] = torch.from_numpy(emb) # (n, 512)
        if self.omni_emb is not None:
            emb_ = np.zeros((self.num_classes, 3584), dtype=np.float32)
            for i in range(video_item['n_cap']):
                emb_[i] = self.omni_emb['%s#%d' % (vid_id, i)]
            data_dict['omni_emb'] = torch.from_numpy(emb_) # (n, 3584)
        data_dict['cap_pool_idx'] = torch.from_numpy(pidx) # (n,)
        data_dict['cap_text'] = texts
        # original boundaries in seconds, only used for scoring
        data_dict['segments_sec'] = video_item['segments'].tolist()
        return data_dict
