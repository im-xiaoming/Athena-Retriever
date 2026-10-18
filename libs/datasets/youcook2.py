from __future__ import annotations
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset, make_generator
from .data_utils import truncate_feats, label_points

from typing import Literal


@register_dataset("youcook2")
class YouCook2Dataset(Dataset):
    def __init__(
        self,
        is_training,      # True: tập train (có cắt ngẫu nhiên và pool caption)
        split: Literal['training', 'validation'],            # list subset cần lấy, ví dụ ['training']
        feat_folder,      # thư mục feature, mỗi video một file <id>.npz
        json_file,        # file json nhãn
        caption_emb_file, # file vector và text của caption
        omni_emb_file,
        default_fps,      # fps quy ước (16), để đổi giây <-> bước
        max_seq_len,      # số bước của mỗi video (256)
        trunc_thresh,     # một đoạn bị cắt chỉ được giữ nếu còn ít nhất tỉ lệ này (0.5)
        crop_ratio,       # (0.9, 1.0): cắt ngẫu nhiên giữ lại 90-100% video
        num_classes: int,      # số caption tối đa trong một video (16)
        scale_factor,     # hệ số giảm độ dài giữa hai tầng pyramid
        regression_range, # khoảng hồi quy của từng tầng pyramid
        backbone_arch,    # (#conv, #stem, #nhánh pyramid), dùng để biết số tầng
    ):
        self.feat_folder = feat_folder
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.num_classes = num_classes

        # vector và text của caption, key là "<video_id>#<chỉ số đoạn>"
        # ['keys', 'emb', 'sentences', 'other', 'mean']
        # emb = (11594, 512)
        z = np.load(caption_emb_file, allow_pickle=True)
        t = np.load(omni_emb_file, allow_pickle=True)
        
        self.cap_emb = {k: v for k, v in zip(z['keys'], z['emb'])}
        self.cap_text = {k: str(s) for k, s in zip(z['keys'], z['sentences'])}
        self.omni_emb = {k.replace('__text', ''): np.array(v) for k, v in t.items()}

        # đọc file nhãn, lấy các video thuộc subset
        self.data_list = self._load_json_db(json_file)
        if is_training:
            self._build_pool()

        # bộ sinh điểm cho pyramid
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
            if any('%s#%d' % (key, i) not in self.omni_emb for i in range(len(anns))):
                continue
            if not os.path.exists(os.path.join(self.feat_folder, key + '.npz')):
                continue
            segments = np.array([a['segment'] for a in anns], dtype=np.float32) # (N, 2)
            dict_db += ({'id': key,
                         'fps': self.default_fps,
                         'duration': value['duration'], # None
                         'segments': segments, # (n, 2)
                         'labels': np.arange(len(anns), dtype=np.int64),   # chỉ số caption; (n,)
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
                if t not in self.pool_index:   # câu mới gặp lần đầu: thêm vào pool
                    self.pool_index[t] = len(self.pool_text)
                    self.pool_text.append(t)
                    emb.append(self.cap_emb[k])
                    omi_emb_.append(self.omni_emb[k])
                self.cap_pool_idx[k] = self.pool_index[t]
        self.pool_emb = np.stack(emb).astype(np.float32) # (N, 512)
        self.pool_omni_emb = np.stack(omi_emb_).astype(np.float32) # (N, 3584)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        vid_id = video_item['id']

        # đọc feature, chuẩn hóa L2 từng hàng
        z = np.load(os.path.join(self.feat_folder, vid_id + '.npz'))
        fv = z['v768'].astype(np.float32) # TxC
        fa = z['a768'].astype(np.float32) # TxC
        fv = fv / np.linalg.norm(fv, axis=1, keepdims=True)
        fa = fa / np.linalg.norm(fa, axis=1, keepdims=True)
        t = min(fv.shape[0], fa.shape[0]) # T

        # Lưới thời gian. n hàng feature (mỗi giây một hàng = default_fps frame) được nội suy về max_seq_len bước,
        # nên một bước dài feat_stride frame và tâm của bước i ở frame (i + 0.5) * feat_stride.
        # Thời điểm t (giây) nằm ở vị trí t * fps / feat_stride - 0.5 trên lưới;
        # model đổi ngược lại bằng (vị trí * feat_stride + 0.5 * num_frames) / fps.
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
            # bỏ các bước GT bắt đầu sau khi feature đã hết; bước bị cắt ở cuối chỉ được giữ nếu còn
            # ít nhất trunc_thresh độ dài; luôn giữ lại ít nhất một bước
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

        # lúc train thì cắt ngẫu nhiên feature
        if self.is_training:
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio)

        # nhãn train cho mọi điểm của pyramid
        points = self.point_generator(self.fpn_strides, data_dict['feats']['visual'], self.is_training)
        
        data_dict['gt_cls_labels'], data_dict['gt_offsets'] = label_points(
            points, data_dict['segments'], data_dict['labels'], self.num_classes)
        data_dict['points'] = points

        # ma trận caption của video, padding cho đủ NMAX
        emb = np.zeros((self.num_classes, 512), dtype=np.float32)
        emb_ = np.zeros((self.num_classes, 3584))
        pidx = np.full((self.num_classes,), -1, dtype=np.int64)
        texts = []
        for i in range(video_item['n_cap']): # n
            k = '%s#%d' % (vid_id, i)
            emb[i] = self.cap_emb[k]
            emb_[i] = self.omni_emb[k]
            texts.append(self.cap_text[k])
            if self.is_training:
                pidx[i] = self.cap_pool_idx[k]
        data_dict['cap_emb'] = torch.from_numpy(emb) # (n, 512)
        data_dict['omni_emb'] = torch.from_numpy(emb_) # (n, 3584)
        data_dict['cap_pool_idx'] = torch.from_numpy(pidx) # (n,)
        data_dict['cap_text'] = texts
        # biên gốc tính bằng giây, chỉ dùng để chấm điểm
        data_dict['segments_sec'] = video_item['segments'].tolist()
        return data_dict
