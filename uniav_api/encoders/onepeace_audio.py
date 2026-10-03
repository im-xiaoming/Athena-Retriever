"""ONE-PEACE audio encoder in pure PyTorch (no fairseq): 1 s of 16 kHz audio -> 1536-d vector.

Port of one_peace/models/adapter/audio.py (AudioAdapter), the audio path of the fusion
encoder, and one_peace_retrieval's audio head:
  layer-normed waveform -> 7-layer conv feature extractor (wav2vec2 style, 49 frames per s)
  -> LayerNorm -> Linear 512->1536 -> + conv positional embedding, CLS prepended
  -> 40 fusion layers (shared attention, audio_ffn) with a shared relative position bias
  -> audio_layer_norm -> CLS -> audio_proj -> L2 normalise.
Hyper-parameters come from the cfg stored in one-peace-audio.pt:
  feature_encoder_spec [(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2, conv_pos_depth 5,
  conv_pos_width 95, conv_pos_groups 16, bucket_size 512, one relative-position table.
Windows follow the training extraction: 1 s windows, 0.5 s hop, each layer-normed alone.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .onepeace_common import Layer, build_and_load, layer_hparams, load_checkpoint

LAYERS = 'encoder_wrapper.fusion_model.layers.'
ADAPTER = 'encoder_wrapper.audio_adapter.'
FEATURE_SPEC = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2


class TransposeLast(nn.Module):
    def forward(self, x):
        return x.transpose(-2, -1)


class ConvFeatureExtraction(nn.Module):
    """Raw waveform (B, S) -> (B, C, T); each block: Conv1d -> Dropout -> LayerNorm over C -> GELU."""

    def __init__(self, spec):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_d = 1
        for dim, k, stride in spec:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_d, dim, k, stride=stride, bias=False), nn.Identity(),
                nn.Sequential(TransposeLast(), nn.LayerNorm(dim), TransposeLast()), nn.GELU()))
            in_d = dim

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class AudioAdapter(nn.Module):
    def __init__(self, dim, heads, conv_pos_depth=5, conv_pos_width=95, conv_pos_groups=16):
        super().__init__()
        feat_dim = FEATURE_SPEC[-1][0]
        self.embed_audios = nn.Sequential(ConvFeatureExtraction(FEATURE_SPEC), TransposeLast(),
                                          nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim))
        k = max(3, conv_pos_width // conv_pos_depth)
        assert k % 2 == 1, 'SamePad trimming for even kernels is not ported (this checkpoint uses k=19)'
        self.embed_positions = nn.Sequential(TransposeLast(), *[
            nn.Sequential(nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2, groups=conv_pos_groups),
                          nn.Identity(), TransposeLast(), nn.LayerNorm(dim, elementwise_affine=False),
                          TransposeLast(), nn.GELU())
            for _ in range(conv_pos_depth)], TransposeLast())
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_embedding = nn.Parameter(torch.zeros(1, dim))   # unused at inference, kept for loading
        self.register_buffer('rp_bucket', torch.zeros(1024, 1024, dtype=torch.long))
        self.rel_pos_table_list = nn.ModuleList([nn.Embedding(2 * 512 + 2, heads)])

    def forward(self, wav):
        bsz = wav.size(0)
        x = self.embed_audios(wav)                                   # (B, T, D)
        pos = torch.cat([self.cls_pos_embed.expand(bsz, -1, -1), self.embed_positions(x)], dim=1)
        x = torch.cat([self.cls_embedding.expand(bsz, -1, -1), x], dim=1) + pos
        seq_len = x.size(1)
        rp = self.rp_bucket[:seq_len, :seq_len]
        bias = self.rel_pos_table_list[0](rp).unsqueeze(0).expand(bsz, -1, -1, -1).permute(0, 3, 1, 2)
        return x, bias


class OnePeaceAudio(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.heads = hp['heads']
        self.audio_adapter = AudioAdapter(hp['dim'], hp['heads'])
        self.layers = nn.ModuleList([
            Layer(hp['dim'], hp['ffn_dim'], hp['heads'], hp['scale_attn'], hp['scale_fc'], hp['scale_heads'],
                  hp['magneto'], hp['layer_scale'], 'audio_ffn') for _ in range(hp['layers'])])
        self.audio_layer_norm = nn.LayerNorm(hp['dim'])
        self.audio_proj = nn.Linear(hp['dim'], hp['dim'])

    def forward(self, wav):
        x, bias = self.audio_adapter(wav)
        x = x.transpose(0, 1)
        bias = bias.contiguous()
        for layer in self.layers:
            x = layer(x, bias)
        x = self.audio_layer_norm(x)
        return F.normalize(self.audio_proj(x[0]), dim=1)


def _remap(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith(ADAPTER):
            out['audio_adapter.' + k[len(ADAPTER):]] = v
        elif k.startswith(LAYERS):
            out['layers.' + k[len(LAYERS):]] = v
        elif k.startswith('encoder_wrapper.fusion_model.audio_layer_norm.'):
            out['audio_layer_norm.' + k.split('.')[-1]] = v
        elif k.startswith('audio_proj.'):
            out[k] = v
    return out


class AudioEncoder:
    """Waveform (16 kHz) -> (n_windows, 1536): 1 s windows every `hop` s, as in training."""

    SR = 16000

    def __init__(self, checkpoint, device, dtype):
        ck = load_checkpoint(checkpoint)
        sd = ck['model'] if 'model' in ck else ck['state_dict']
        sd = {k: v for k, v in sd.items()
              if k.startswith((ADAPTER, 'encoder_wrapper.fusion_model.audio_layer_norm.', 'audio_proj.'))
              or (k.startswith(LAYERS) and not k.split('.', 4)[4].startswith(('image_ffn', 'text_ffn')))}
        hp = layer_hparams(sd, LAYERS, 'audio_ffn')
        hp['heads'] = sd[ADAPTER + 'rel_pos_table_list.0.weight'].shape[1]
        self.device, self.dtype = device, dtype
        self.model = build_and_load(lambda: OnePeaceAudio(hp), _remap(sd), device, dtype)

    @torch.no_grad()
    def __call__(self, wav, hop=0.5, batch=32):
        window, step = self.SR, int(self.SR * hop)
        wav = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        if wav.numel() < window:   # shorter than 1 s: repeat, as process_audio does
            wav = wav.repeat(math.ceil(window / max(wav.numel(), 1)))[:window]
        n = max(0, (wav.numel() - window) // step) + 1
        chunks = F.layer_norm(wav.unfold(0, window, step)[:n], (window,))
        out = []
        for i in range(0, n, batch):
            out.append(self.model(chunks[i:i + batch].to(self.device, self.dtype)).float().cpu())
        return torch.cat(out).numpy()
