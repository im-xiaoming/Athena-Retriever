"""Paths and runtime settings. Every field can be overridden with an environment variable
ATHENA_<FIELD> (upper case), e.g. ATHENA_DEVICE=cpu or ATHENA_CHECKPOINT=/path/best_cap.pth.tar.
"""
import os
from dataclasses import dataclass, fields

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # API checkpoint from tools/export_api_ckpt.py; it carries its own training config.
    # A raw training checkpoint (ckpt/<run>/best_cap.pth.tar) is read with model_config instead.
    checkpoint: str = os.path.join(ROOT, 'ckpt', 'api', 'athena.pth')
    model_config: str = os.path.join(ROOT, 'configs', 'youcook2_event.yaml')
    samples_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
    caption_pool: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'caption_pool.npz')
    # InternVideo2 encoders: video + audio for describe_video; the video checkpoint also holds the
    # text tower used for search
    iv2_video_encoder: str = os.path.join(ROOT, 'ckpt', 'internvideo2', 'InternVideo2-stage2_1b-224p-f4.pt')
    iv2_audio_encoder: str = os.path.join(ROOT, 'ckpt', 'internvideo2', 'audio_6b.pth')
    iv2_repo: str = os.path.join(ROOT, 'InternVideo')   # git clone --depth 1 https://github.com/OpenGVLab/InternVideo
    index_dir: str = os.path.join(ROOT, 'athena', 'index')
    feature_cache: str = ''         # folder to cache encoder features per video ('' = off)
    device: str = 'auto'            # auto | cuda | mps | cpu
    keep_encoders_loaded: str = 'auto'   # auto: keep on GPUs with >= 10 GB, otherwise load per use
    # event selection, calibrated on the validation set (see athena/calibrate.py)
    min_score: float = 0.40          # athena.pth (best_seg weights), val: F1@IoU0.5 0.535
    max_overlap: float = 0.3        # drop an event overlapping a better one by more than this IoU
    max_events: int = 30
    alternatives: int = 3           # extra caption candidates returned per event
    # caption source: 'retrieve' picks a train caption (default); 'generate' writes one with GPT-2
    # (athena/generator.py); the retrieved caption is then kept as 'retrieved_caption'
    caption_mode: str = 'retrieve'
    # checkpoints with state_dict_seg (export_api_ckpt.py): segment and ground with those weights (the
    # best segmentation epoch) and caption with the main ones; False uses the main weights for everything
    use_seg_weights: bool = True
    generator: str = os.path.join(ROOT, 'ckpt', 'api', 'capgen_prefix.pth')

    @classmethod
    def from_env(cls, **kw):
        """Config(**kw), then every ATHENA_<FIELD> environment variable on top (converted to the
        field's type)."""
        cfg = cls(**kw)
        for f in fields(cls):
            v = os.environ.get('ATHENA_' + f.name.upper())
            if v is not None:
                t = type(getattr(cfg, f.name))
                # bool('False') is True: parse booleans by their text
                setattr(cfg, f.name, v.strip().lower() in ('1', 'true', 'yes') if t is bool else t(v))
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
    """Whether the InternVideo2 encoders stay in memory between videos (Config.keep_encoders_loaded)."""
    if cfg.keep_encoders_loaded != 'auto':
        return cfg.keep_encoders_loaded.lower() in ('1', 'true', 'yes')
    if device.type == 'cuda':
        return torch.cuda.get_device_properties(device).total_memory >= 10e9   # both fp16 encoders: ~6.9 GB peak
    return False   # MPS shares system memory and CPU holds fp32: load the big encoders per use
