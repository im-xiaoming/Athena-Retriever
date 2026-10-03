"""Dense InternVideo2 features for uncut YouCook2 videos: one video and one audio vector per second.

  python tools/extract_internvideo2.py --videos /content/yc2_videos --out /content/iv2_feats \\
      --repo /content/InternVideo --video-ckpt /content/iv2_ckpt/InternVideo2-stage2_1b-224p-f4.pt \\
      --audio-ckpt /content/iv2_ckpt/audio_6b.pth

Video: InternVideo2-Stage2 1B. Window i covers about [i - 0.5, i + 1.5) s with 4 frames
sampled at 2 fps and resized to 224x224 (as in the official demo). Outputs per window:
  v768 : pooled vision embedding (attention-pooled, before projection)
  v512 : vision_proj(v768), L2-normalised, in the space shared with text
Audio: BEATs encoder from InternVideo2-Stage2-6B-Audio, 16 kHz mono, a 3 s window
centred on each second, mean-pooled to
  a768 : audio embedding
Each video becomes <out>/<video_id>.npz (float16) with n_sec rows. Existing outputs are
skipped, so the script resumes. Decoding runs in a thread pool ahead of the GPU.
"""
import argparse
import os
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def stub_flash_attn():
    """InternVideo2 imports flash_attn at module level; we run with flash attention disabled,
    so its FlashAttention wrapper is replaced by a placeholder that is never called."""
    m = types.ModuleType('models.backbones.internvideo2.flash_attention_class')

    class FlashAttention(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError('flash attention is disabled')

    m.FlashAttention = FlashAttention
    sys.modules[m.__name__] = m


def register_packages(repo):
    """Register InternVideo2's packages as empty namespaces so importing one backbone file
    does not run the package __init__ files, which import every model (and flash_attn)."""
    base = os.path.join(repo, 'InternVideo2', 'multi_modality', 'models')
    for name, path in (('models', base), ('models.backbones', os.path.join(base, 'backbones')),
                       ('models.backbones.internvideo2', os.path.join(base, 'backbones', 'internvideo2')),
                       ('models.backbones.beats', os.path.join(base, 'backbones', 'beats'))):
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m


# BEATs iter3 configuration, copied from the cfg stored in BEATs_iter3_plus_AS2M.pt; the
# encoder weights in audio_6b.pth match it exactly. finetuned_model=False: no AudioSet head.
BEATS_ITER3_CFG = {
    'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072,
    'encoder_attention_heads': 12, 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0,
    'activation_dropout': 0.0, 'encoder_layerdrop': 0.0, 'dropout_input': 0.0, 'layer_norm_first': False,
    'conv_bias': False, 'conv_pos': 128, 'conv_pos_groups': 16, 'relative_position_embedding': True,
    'num_buckets': 320, 'max_distance': 800, 'gru_rel_pos': True, 'deep_norm': True, 'input_patch_size': 16,
    'layer_wise_gradient_decay_ratio': 0.6, 'embed_dim': 512, 'finetuned_model': False,
    'predictor_dropout': 0.0, 'predictor_class': 527,
}


class EasyDict(dict):
    __getattr__ = dict.get


def build_video_model(repo, ckpt, device):
    register_packages(repo)
    stub_flash_attn()
    from models.backbones.internvideo2.internvideo2 import pretrain_internvideo2_1b_patch14_224
    cfg = EasyDict(vision_encoder=EasyDict(
        clip_embed_dim=768, use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False,
        num_frames=4, tubelet_size=1, sep_image_video_pos_embed=True, use_checkpoint=False,
        checkpoint_num=0, clip_teacher_embed_dim=3200, clip_teacher_final_dim=768,
        clip_norm_type='l2', clip_return_layer=6, clip_student_return_interval=1, pretrained=None))
    enc = pretrain_internvideo2_1b_patch14_224(cfg)
    proj = torch.nn.Linear(768, 512)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('module', sd.get('model', sd))
    venc = {k[len('vision_encoder.'):]: v for k, v in sd.items() if k.startswith('vision_encoder.')}
    vproj = {k[len('vision_proj.'):]: v for k, v in sd.items() if k.startswith('vision_proj.')}
    missing, unexpected = enc.load_state_dict(venc, strict=False)
    real_missing = [k for k in missing if 'clip_decoder' not in k and 'final_clip_decoder' not in k]
    print('video encoder: %d tensors loaded, missing %d (%s), unexpected %d'
          % (len(venc), len(real_missing), real_missing[:5], len(unexpected)), flush=True)
    proj.load_state_dict(vproj)
    enc = enc.to(device).half().eval()
    proj = proj.to(device).half().eval()
    return enc, proj


def build_audio_model(repo, ckpt, device):
    register_packages(repo)
    from models.backbones.beats.BEATs import BEATs, BEATsConfig
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = ck['model'] if 'model' in ck else ck   # audio_6b.pth is a bare state dict without 'cfg'
    model = BEATs(BEATsConfig(BEATS_ITER3_CFG))
    print('audio encoder:', model.load_state_dict(sd, strict=True), flush=True)
    return model.to(device).eval()


def ffmpeg_frames(path, fps=2, size=224):
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-vf', 'fps=%d,scale=%d:%d' % (fps, size, size),
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)


def ffmpeg_audio(path, sr=16000):
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32)


def decode(path):
    frames = ffmpeg_frames(path)
    wav = ffmpeg_audio(path)
    return frames, wav


@torch.no_grad()
def video_features(enc, proj, frames, device, batch=32):
    n_sec = max(1, int(np.ceil(len(frames) / 2)))
    last = len(frames) - 1
    idx = np.clip(np.arange(n_sec)[:, None] * 2 + np.array([-1, 0, 1, 2])[None], 0, last)   # (n_sec, 4)
    v768, v512 = [], []
    for s in range(0, n_sec, batch):
        clip = (frames[idx[s:s + batch]].astype(np.float32) / 255.0 - MEAN) / STD           # (B,4,H,W,3)
        x = torch.from_numpy(clip).permute(0, 4, 1, 2, 3).to(device).half()                 # (B,3,4,H,W)
        # x_vis_only skips the CLIP-distillation decoders, which only matter in training
        pooled = enc.clip_projector(enc(x, None, False, x_vis_only=True)).reshape(len(clip), -1)
        v768.append(pooled.float().cpu())
        v512.append(F.normalize(proj(pooled).float(), dim=-1).cpu())
    return torch.cat(v768).numpy(), torch.cat(v512).numpy()


@torch.no_grad()
def audio_features(model, wav, n_sec, device, sr=16000, batch=32):
    pad = np.pad(wav, (sr, 2 * sr))                          # window i = [i-1, i+2) s
    win = np.stack([pad[i * sr:(i + 3) * sr] for i in range(n_sec)])
    out = []
    for s in range(0, n_sec, batch):
        x = torch.from_numpy(win[s:s + batch]).to(device)
        feats, _ = model.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
        out.append(feats.mean(dim=1).float().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--videos', required=True); p.add_argument('--out', required=True)
    p.add_argument('--repo', required=True); p.add_argument('--video-ckpt', required=True)
    p.add_argument('--audio-ckpt', required=True)
    p.add_argument('--limit', type=int, default=0, help='only the first N videos (for timing)')
    p.add_argument('--workers', type=int, default=6, help='decode threads ahead of the GPU')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cuda')
    enc, proj = build_video_model(a.repo, a.video_ckpt, dev)
    beats = build_audio_model(a.repo, a.audio_ckpt, dev)
    todo = sorted(f for f in os.listdir(a.videos) if f.endswith('.mp4')
                  and not os.path.exists(os.path.join(a.out, f[:-4] + '.npz')))
    if a.limit:
        todo = todo[:a.limit]
    print('%d videos to do' % len(todo), flush=True)
    t0, secs = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        ahead = a.workers                  # bounded prefetch: decoded videos are ~200 MB each
        futures = {i: ex.submit(decode, os.path.join(a.videos, todo[i])) for i in range(min(ahead, len(todo)))}
        for n, f in enumerate(todo, 1):
            frames, wav = futures.pop(n - 1).result()
            if n - 1 + ahead < len(todo):
                futures[n - 1 + ahead] = ex.submit(decode, os.path.join(a.videos, todo[n - 1 + ahead]))
            if len(frames) == 0:
                print('SKIP %s: no frames' % f, flush=True); continue
            v768, v512 = video_features(enc, proj, frames, dev)
            a768 = audio_features(beats, wav, len(v768), dev) if len(wav) else np.zeros((len(v768), 768), np.float32)
            np.savez(os.path.join(a.out, f[:-4] + '.npz'), v768=v768.astype(np.float16),
                     v512=v512.astype(np.float16), a768=a768.astype(np.float16))
            secs += len(v768)
            el = time.time() - t0
            print('%s %d/%d %s: %d s | %.1f video-s/s | ETA %.0f min' % (
                time.strftime('%H:%M:%S'), n, len(todo), f, len(v768), secs / el,
                (len(todo) - n) * el / n / 60), flush=True)
    print('EXTRACT_DONE', flush=True)


if __name__ == '__main__':
    main()
