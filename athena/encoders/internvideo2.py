"""InternVideo2 audio-visual feature encoder: video file -> (visual, audio) features, as in training.

Reuses the model builders and the decoding of tools/extract_internvideo2.py, the script that made
the training features, so both paths run the same code. Needs the upstream repo cloned unmodified
at <repo>/InternVideo (git clone --depth 1 https://github.com/OpenGVLab/InternVideo) and timm,
einops and torchaudio.

visual: 2 fps, 224x224, one 4-frame window per second -> InternVideo2-1B pooled vision v768
        and/or its text-aligned projection v512 (training option iv2_video_keys)
audio : 16 kHz mono, 3 s window centred on each second -> BEATs mean -> a768
Rows are L2-normalised when the training used iv2_l2norm, exactly as the dataset does.
"""
import importlib.util
import os

import numpy as np

from .media import find_ffmpeg, probe

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _extract_module():
    spec = importlib.util.spec_from_file_location(
        'extract_internvideo2', os.path.join(ROOT, 'tools', 'extract_internvideo2.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


class InternVideo2AVEncoder:
    name = 'iv2'
    dim_audio = 768
    feat_stride, num_frames, fps = 16, 16, 16   # one row per second, centred on i + 0.5 s

    def __init__(self, video_ckpt, audio_ckpt, repo, device, video_keys=('v768',), l2norm=True,
                 keep_loaded=True):
        self.video_ckpt, self.audio_ckpt, self.repo = video_ckpt, audio_ckpt, repo
        self.device, self.keep_loaded = device, keep_loaded
        self.video_keys, self.l2norm = list(video_keys), l2norm
        self.dim_visual = sum(512 if k == 'v512' else 768 for k in self.video_keys)
        self.x = _extract_module()
        self.ffmpeg = find_ffmpeg()
        self._video = self._audio = None

    def _release(self, which):
        if not self.keep_loaded:
            setattr(self, which, None)
            if self.device.type == 'cuda':
                import torch
                torch.cuda.empty_cache()

    def encode_raw(self, path):
        """The arrays tools/extract_internvideo2.py stores: v768, v512, a768 (float32)."""
        frames, wav = self.x.decode(path, ffmpeg=self.ffmpeg)
        if len(frames) == 0:
            raise RuntimeError('no video frames decoded from %s' % path)
        if self._video is None:
            self._video = self.x.build_video_model(self.repo, self.video_ckpt, self.device)
        v768, v512 = self.x.video_features(*self._video, frames, self.device)
        self._release('_video')
        n = len(v768)
        if len(wav):
            if self._audio is None:
                self._audio = self.x.build_audio_model(self.repo, self.audio_ckpt, self.device)
            a768 = self.x.audio_features(self._audio, wav, n, self.device)
            self._release('_audio')
        else:
            a768 = np.zeros((n, 768), np.float32)
        return {'v768': v768, 'v512': v512, 'a768': a768, 'has_audio': bool(len(wav) and wav.any()),
                'duration': probe(path, self.ffmpeg)['duration'] or len(frames) / 2.0}

    def to_training_format(self, raw):
        """Same transform as libs/datasets/youcook2_cap.py applies to a stored .npz."""
        parts = [np.asarray(raw[k], np.float32) for k in self.video_keys]
        audio = np.asarray(raw['a768'], np.float32)
        if self.l2norm:
            parts, audio = [_l2(p) for p in parts], _l2(audio)
        return np.concatenate(parts, axis=1), audio

    def encode(self, path):
        raw = self.encode_raw(path)
        visual, audio = self.to_training_format(raw)
        return {'visual': visual, 'audio': audio, 'duration': raw['duration'], 'has_audio': raw['has_audio']}
