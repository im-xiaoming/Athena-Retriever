"""YouCook2 dataset for event segmentation + captioning.

Differs from anet.py in three ways:
  - each segment's label is the CAPTION INDEX within the video (0..N-1), not a global class id
  - every sample carries its embedded captions (NMAX, 1536) and a validity mask (NMAX,)
  - num_classes = NMAX, the maximum number of captions in one video
so the existing label_points produces gt_cls_labels (P, NMAX) unchanged.
"""
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset, make_generator
from .data_utils import truncate_feats, label_points

NMAX = 16   # maximum number of captions in one YouCook2 video


@register_dataset("youcook2_cap")
class YouCook2CaptionDataset(Dataset):
    def __init__(
        self, is_training, split, feat_folder, json_file, caption_emb_file,
        feat_stride, num_frames, default_fps, downsample_rate, max_seq_len,
        max_buffer_len_factor, scale_factor, regression_range, backbone_arch,
        class_aware, trunc_thresh, crop_ratio, num_classes, file_prefix,
        file_ext, force_upsampling, multi_modal, omni_emb_file=None,
    ):
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert os.path.exists(caption_emb_file)
        assert num_classes == NMAX, 'num_classes phai bang %d' % NMAX
        self.multi_modal = multi_modal
        self.feat_folder = feat_folder
        self.file_prefix = file_prefix or ''
        self.file_ext = file_ext
        self.json_file = json_file
        self.force_upsampling = force_upsampling
        self.split = split
        self.is_training = is_training
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.crop_ratio = crop_ratio

        # caption embeddings, keyed "<video_id>#<segment index>"
        z = np.load(caption_emb_file, allow_pickle=True)
        self.cap_emb = {k: v for k, v in zip(z['keys'], z['emb'])}
        self.cap_text = {k: s for k, s in zip(z['keys'], z['sentences'])}
        self.emb_dim = z['emb'].shape[1]

        self.data_list = self._load_json_db(json_file)
        self._build_pool()
        self.omni = bool(omni_emb_file) and is_training   # only the train split needs the teacher
        if self.omni:
            self._load_omni(omni_emb_file)
        self.db_attributes = {
            'dataset_name': 'YouCook2 caption grounding',
            'tiou_thresholds': np.array([0.3, 0.5, 0.7]),
            'empty_label_ids': [],
        }

        self.fpn_strides = [scale_factor ** i for i in range(backbone_arch[-1] + 1)]
        self.reg_range = regression_range
        self.class_aware = class_aware
        self.max_div_factor = max(self.fpn_strides)
        for s in self.fpn_strides:
            assert max_seq_len % s == 0, 'max_seq_len phai chia het cho fpn stride'
        self.point_generator = make_generator('point', **{
            'max_seq_len_ori': self.max_seq_len,
            'max_buffer_len_factor': max_buffer_len_factor,
            'fpn_levels': len(self.fpn_strides),
            'scale_factor': scale_factor,
            'regression_range': self.reg_range,
            'max_div_factor': self.max_div_factor,
        })

    def _build_pool(self):
        """Unique caption pool of this split; identical captions share one index.

        Only the train pool is used: as negatives during training and as the pool
        captions are picked from at inference.
        """
        self.pool_text, self.pool_index, emb = [], {}, []
        self.cap_pool_idx = {}
        for it in self.data_list:
            for i in range(it['n_cap']):
                k = '%s#%d' % (it['id'], i); t = str(self.cap_text[k])
                if t not in self.pool_index:
                    self.pool_index[t] = len(self.pool_text)
                    self.pool_text.append(t); emb.append(self.cap_emb[k])
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
        av, self.cap_av_idx = [], {}
        for it in self.data_list:
            for i in range(it['n_cap']):
                k = '%s#%d' % (it['id'], i)
                j = self.cap_pool_idx[k]
                if not self.omni_text_ok[j] and k + '__text' in have:
                    txt[j] = z[k + '__text']; self.omni_text_ok[j] = True
                if k + '__av' in have:
                    self.cap_av_idx[k] = len(av); av.append(z[k + '__av'])
        self.omni_text_pool = txt
        self.omni_av_pool = (np.stack(av) if av else
                             np.zeros((0, self.omni_dim), dtype=np.float16))
        print('Omni teacher : %d/%d caption vectors, %d clip vectors from %s' % (
            self.omni_text_ok.sum(), len(self.pool_text), len(av), path), flush=True)

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
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

    def __getitem__(self, idx):
        item = self.data_list[idx]
        vid = item['id']
        pre = os.path.join(self.feat_folder, self.file_prefix + vid)
        fv = np.load(pre + '_one_peace_video_finetune' + self.file_ext).astype(np.float32)
        fa = np.load(pre + '_one_peace_audio' + self.file_ext).astype(np.float32)
        n = min(fv.shape[0], fa.shape[0])
        fv, fa = fv[:n], fa[:n]

        # recompute the time scale (same as case 2 in anet.py)
        feat_stride = float((n - 1) * self.feat_stride + self.num_frames) / self.max_seq_len
        num_frames = feat_stride
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
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset,
                self.crop_ratio, self.multi_modal)

        points = self.point_generator(self.fpn_strides, data_dict['feats']['visual'], self.is_training)
        data_dict['gt_cls_labels'], data_dict['gt_offsets'] = label_points(
            points, data_dict['segments'], data_dict['labels'], self.num_classes, self.class_aware)
        data_dict['points'] = points

        # caption matrix of the video, padded to NMAX
        emb = np.zeros((NMAX, self.emb_dim), dtype=np.float32)
        cmask = np.zeros((NMAX,), dtype=bool)
        texts = [''] * NMAX
        pidx = np.full((NMAX,), -1, dtype=np.int64)
        for i in range(item['n_cap']):
            k = '%s#%d' % (vid, i)
            emb[i] = self.cap_emb[k]; cmask[i] = True; texts[i] = str(self.cap_text[k])
            pidx[i] = self.cap_pool_idx[k]
        data_dict['cap_emb'] = torch.from_numpy(emb)
        data_dict['cap_mask'] = torch.from_numpy(cmask)
        data_dict['cap_pool_idx'] = torch.from_numpy(pidx)
        if self.omni:
            aidx = np.full((NMAX,), -1, dtype=np.int64)
            for i in range(item['n_cap']):
                aidx[i] = self.cap_av_idx.get('%s#%d' % (vid, i), -1)
            data_dict['cap_av_idx'] = torch.from_numpy(aidx)
        data_dict['cap_text'] = texts
        # original boundaries in seconds, used only for scoring
        data_dict['segments_sec'] = item['segments'].tolist()
        return data_dict
