"""ONE-PEACE audio-visual feature encoder: video file -> (visual, audio) features, as in training.

visual: 16 fps, 256x256 centre crop, 16-frame windows every 8 frames (0.5 s), mean CLS of the
        K400-fine-tuned video backbone -> (T, 1536)       = <id>_one_peace_video_finetune.npy
audio : 16 kHz mono, 1 s windows every 0.5 s -> ONE-PEACE audio CLS -> audio_proj -> L2
        -> (T, 1536)                                      = <id>_one_peace_audio.npy
"""
import numpy as np
import torch

from .base import FeatureEncoder
from .media import find_ffmpeg, iter_clip_chunks, iter_video_frames, load_audio, probe
from .onepeace_audio import AudioEncoder
from .onepeace_video import IMG_MEAN, IMG_STD, load_k400_checkpoint

FPS, NUM_FRAMES, STRIDE = 16, 16, 8


class VideoEncoder:
    def __init__(self, checkpoint, device, dtype):
        self.device, self.dtype = device, dtype
        use_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.model = load_k400_checkpoint(checkpoint, use_sdpa=use_sdpa).to(device=device, dtype=dtype).eval()
        self.mean = torch.tensor(IMG_MEAN, device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(IMG_STD, device=device).view(1, 3, 1, 1, 1)

    @torch.no_grad()
    def __call__(self, path, ffmpeg, info, batch=8):
        out = []
        frames = iter_video_frames(path, fps=FPS, size=256, ffmpeg=ffmpeg, info=info)
        for chunk, m in iter_clip_chunks(frames, NUM_FRAMES, STRIDE, batch):
            x = torch.from_numpy(chunk).to(self.device)
            clips = x.unfold(0, NUM_FRAMES, STRIDE)[:m].permute(0, 3, 4, 1, 2)   # m, 3, T, H, W
            x = ((clips.float() - self.mean) / self.std).to(self.dtype)
            out.append(self.model.extract_clip_features(x).float().cpu())
        return torch.cat(out).numpy()


class OnePeaceAVEncoder(FeatureEncoder):
    name = 'onepeace'
    dim_visual = dim_audio = 1536
    feat_stride, num_frames, fps = STRIDE, NUM_FRAMES, FPS   # time grid of the features

    def __init__(self, video_ckpt, audio_ckpt, device, dtype, keep_loaded=True):
        self.video_ckpt, self.audio_ckpt = video_ckpt, audio_ckpt
        self.device, self.dtype, self.keep_loaded = device, dtype, keep_loaded
        self.ffmpeg = find_ffmpeg()
        self._video = self._audio = None

    def _video_encoder(self):
        if self._video is None:
            self._video = VideoEncoder(self.video_ckpt, self.device, self.dtype)
        return self._video

    def _audio_encoder(self):
        if self._audio is None:
            self._audio = AudioEncoder(self.audio_ckpt, self.device, self.dtype)
        return self._audio

    def _release(self, which):
        """On machines without a big GPU the two 1.5B encoders do not fit at once: free after use."""
        if not self.keep_loaded:
            setattr(self, which, None)
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def encode(self, path):
        info = probe(path, self.ffmpeg)
        visual = self._video_encoder()(path, self.ffmpeg, info)
        self._release('_video')
        wav = load_audio(path, ffmpeg=self.ffmpeg)
        if wav is None:   # no audio track: silence of the same length, as the training extraction did
            wav = np.zeros(int((info['duration'] or 0) * AudioEncoder.SR), np.float32)
        audio = self._audio_encoder()(wav, hop=STRIDE / FPS)
        self._release('_audio')
        n = min(len(visual), len(audio))
        return {'visual': visual[:n], 'audio': audio[:n], 'duration': info['duration'] or n * STRIDE / FPS,
                'has_audio': wav.any() if len(wav) else False}
