"""Paths and runtime settings. Every field can be overridden with an environment variable
UNIAV_<FIELD> (upper case), e.g. UNIAV_DEVICE=cpu or UNIAV_CHECKPOINT=/path/best_cap.pth.tar.
"""
import os
from dataclasses import dataclass, field, fields

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONEPEACE = os.path.join(ROOT, 'ONEPEACE_extract_embd_code', 'models')


@dataclass
class Config:
    checkpoint: str = os.path.join(ROOT, 'ckpt', 'omni65', 'best_cap.pth.tar')
    model_config: str = os.path.join(ROOT, 'configs', 'youcook2_event.yaml')
    caption_pool: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'caption_pool.npz')
    video_encoder: str = os.path.join(ONEPEACE, 'onepeace_video_k400.pth')
    audio_encoder: str = os.path.join(ONEPEACE, 'one-peace-audio.pt')
    text_encoder: str = os.path.join(ONEPEACE, 'one-peace-text.pt')
    index_dir: str = os.path.join(ROOT, 'uniav_api', 'index')
    feature_cache: str = ''         # folder to cache encoder features per video ('' = off)
    device: str = 'auto'            # auto | cuda | mps | cpu
    keep_encoders_loaded: str = 'auto'   # auto: keep on GPUs with >= 10 GB, otherwise load per use
    # event selection, calibrated on the validation set (see uniav_api/calibrate.py)
    min_score: float = 0.40          # val: F1@IoU0.5 0.476, 8.5 events per video (GT 7.7)
    max_overlap: float = 0.3        # drop an event overlapping a better one by more than this IoU
    max_events: int = 30
    alternatives: int = 3           # extra caption candidates returned per event

    @classmethod
    def from_env(cls, **kw):
        cfg = cls(**kw)
        for f in fields(cls):
            v = os.environ.get('UNIAV_' + f.name.upper())
            if v is not None:
                setattr(cfg, f.name, type(getattr(cfg, f.name))(v))
        return cfg


def pick_device(pref='auto'):
    """CUDA (NVIDIA, Windows/Linux) -> MPS (Apple Silicon) -> CPU. fp16 on GPUs, fp32 on CPU."""
    if pref == 'auto':
        if torch.cuda.is_available():
            pref = 'cuda'
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            pref = 'mps'
        else:
            pref = 'cpu'
    device = torch.device(pref)
    dtype = torch.float32 if device.type == 'cpu' else torch.float16
    return device, dtype


def keep_encoders(cfg, device):
    if cfg.keep_encoders_loaded != 'auto':
        return cfg.keep_encoders_loaded.lower() in ('1', 'true', 'yes')
    if device.type == 'cuda':
        return torch.cuda.get_device_properties(device).total_memory >= 10e9   # both fp16 encoders: ~6.9 GB peak
    return False   # MPS shares system memory and CPU holds fp32: load the big encoders per use
