"""Pieces shared by the pure-PyTorch ONE-PEACE text and audio branches (no fairseq).

Ported from ONEPEACE_extract_embd_code/onepeace_text.py, which reproduces fairseq ONE-PEACE
text features to 1.6e-7. The 40 fusion layers share self-attention across modalities and
differ only in their FFN (text_ffn / audio_ffn), so one Layer class serves both.
"""
import importlib.util
import math
import os
import pickle
import re

import torch
import torch.nn.functional as F
from torch import nn

BPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bpe')


class Tokenizer:
    """GPT-2 BPE + fairseq Dictionary, as task.bpe / task.dict in ONE-PEACE."""

    def __init__(self, bpe_dir=BPE_DIR):
        spec = importlib.util.spec_from_file_location('gpt2_bpe_utils', os.path.join(bpe_dir, 'gpt2_bpe_utils.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.bpe = mod.get_encoder(os.path.join(bpe_dir, 'encoder.json'), os.path.join(bpe_dir, 'vocab.bpe'))
        self.symbols = ['<s>', '<pad>', '</s>', '<unk>']
        with open(os.path.join(bpe_dir, 'dict.txt'), encoding='utf-8') as f:
            for line in f:
                self.symbols.append(line.rstrip().rsplit(' ', 1)[0])
        self.index = {s: i for i, s in enumerate(self.symbols)}
        self.pad, self.eos, self.unk = 1, 2, 3

    def __len__(self):
        return len(self.symbols)

    def encode(self, text, max_len=70):
        bpe_str = ' '.join(map(str, self.bpe.encode(' {}'.format(text.lower()))))
        ids = [self.index.get(t, self.unk) for t in re.sub(r'\s+', ' ', bpe_str).strip().split()]
        return ids[:max_len] + [self.eos]

    def __call__(self, texts):
        seqs = [self.encode(t) for t in texts]
        out = torch.full((len(seqs), max(map(len, seqs))), self.pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s)
        return out


def make_token_bucket_position(bucket_size, max_position):
    context_pos = torch.arange(max_position, dtype=torch.long)[:, None]
    memory_pos = torch.arange(max_position, dtype=torch.long)[None, :]
    relative_pos = context_pos - memory_pos
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos))
    log_pos = mid + torch.ceil(torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)).long()
    bucket_pos = torch.where(abs_pos.le(mid), relative_pos, log_pos * sign).long()
    return bucket_pos + bucket_size - 1


class MultiheadAttention(nn.Module):
    def __init__(self, dim, heads, scale_heads, magneto):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.scaling = self.head_dim ** -0.5
        self.c_attn = nn.Parameter(torch.ones(heads)) if scale_heads else None
        self.ln = nn.LayerNorm(dim) if magneto else None
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        T, B, C = x.size()
        q = self.q_proj(x).view(T, B * self.heads, self.head_dim).transpose(0, 1) * self.scaling
        k = self.k_proj(x).view(T, B * self.heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(T, B * self.heads, self.head_dim).transpose(0, 1)
        w = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            w = w + attn_mask.reshape(-1, T, T)
        w = F.softmax(w.float(), dim=-1).type_as(w)
        attn = torch.bmm(w, v).transpose(0, 1).reshape(T, B, C)
        if self.c_attn is not None:
            attn = torch.einsum('nbhd,h->nbhd', attn.view(T, B, self.heads, self.head_dim), self.c_attn).reshape(T, B, C)
        if self.ln is not None:
            attn = self.ln(attn)
        return self.out_proj(attn)


class GeGLU(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.wi_0 = nn.Linear(dim, ffn_dim, bias=False)
        self.wi_1 = nn.Linear(dim, ffn_dim, bias=False)

    def forward(self, x):
        return F.gelu(self.wi_0(x)) * self.wi_1(x)


class Layer(nn.Module):
    """One fusion layer; `ffn_name` is 'text_ffn' or 'audio_ffn' to match the checkpoint keys."""

    def __init__(self, dim, ffn_dim, heads, scale_attn, scale_fc, scale_heads, magneto, layer_scale, ffn_name):
        super().__init__()
        self.ffn_name = ffn_name
        self.self_attn = MultiheadAttention(dim, heads, scale_heads, magneto)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.attn_ln = nn.LayerNorm(dim) if scale_attn else None
        self.final_layer_norm = nn.LayerNorm(dim)
        # same Sequential indices as the original: 0 GeGLU, 1 dropout, 2 LayerNorm, 3 Linear
        setattr(self, ffn_name, nn.Sequential(GeGLU(dim, ffn_dim), nn.Identity(),
                                              nn.LayerNorm(ffn_dim) if scale_fc else nn.Identity(),
                                              nn.Linear(ffn_dim, dim)))
        self.gamma_1 = nn.Parameter(torch.ones(dim)) if layer_scale else None
        self.gamma_2 = nn.Parameter(torch.ones(dim)) if layer_scale else None

    def forward(self, x, attn_bias):
        h = self.self_attn(self.self_attn_layer_norm(x), attn_bias)
        if self.attn_ln is not None:
            h = self.attn_ln(h)
        x = x + (self.gamma_1 * h if self.gamma_1 is not None else h)
        h = getattr(self, self.ffn_name)(self.final_layer_norm(x))
        return x + (self.gamma_2 * h if self.gamma_2 is not None else h)


def layer_hparams(sd, prefix, ffn_name):
    """Shapes and switches of the fusion layers, read from the state dict."""
    n_layers = 1 + max(int(k[len(prefix):].split('.')[0]) for k in sd if k.startswith(prefix))
    L = prefix + '0.'
    return dict(
        layers=n_layers, dim=sd[L + 'self_attn.q_proj.weight'].shape[0],
        ffn_dim=sd[L + ffn_name + '.0.wi_0.weight'].shape[0],
        scale_attn=L + 'attn_ln.weight' in sd, scale_fc=L + ffn_name + '.2.weight' in sd,
        scale_heads=L + 'self_attn.c_attn' in sd, magneto=L + 'self_attn.ln.weight' in sd,
        layer_scale=L + 'gamma_1' in sd)


class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): self.__dict__['_state'] = state


class _LenientUnpickler(pickle.Unpickler):
    """fairseq checkpoints may pickle fairseq/omegaconf objects; replace them with stubs."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


class LenientPickle:
    Unpickler = _LenientUnpickler
    load = pickle.load


def load_checkpoint(path):
    """mmap-load a ONE-PEACE checkpoint: tensors are read from disk on demand, not copied into RAM."""
    try:
        return torch.load(path, map_location='cpu', mmap=True, weights_only=False, pickle_module=LenientPickle)
    except (TypeError, RuntimeError):   # older torch or a non-zip checkpoint
        return torch.load(path, map_location='cpu', pickle_module=LenientPickle)


def build_and_load(make_model, state_dict, device, dtype):
    """Build on the meta device and assign the checkpoint tensors, then move to device/dtype."""
    try:
        with torch.device('meta'):
            model = make_model()
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    except (TypeError, AttributeError, RuntimeError):   # torch < 2.1
        model = make_model()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.endswith('rp_bucket')]
    if missing or unexpected:
        raise RuntimeError('checkpoint mismatch: missing %s, unexpected %s' % (missing[:8], unexpected[:8]))
    if hasattr(model, 'init_buffers'):   # buffers the checkpoint does not store (e.g. rp_bucket)
        model.init_buffers()
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            raise RuntimeError('buffer %s was not loaded' % name)
    return model.to(device=device, dtype=dtype).eval()
