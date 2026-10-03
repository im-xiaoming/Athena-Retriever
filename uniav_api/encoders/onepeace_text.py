"""ONE-PEACE text encoder (pure PyTorch): sentence -> 1536-d L2-normalised vector.

Same computation as ONEPEACE_extract_embd_code/onepeace_text.py and fairseq's
extract_text_features: text_adapter -> 40 fusion layers (text_ffn) -> text_layer_norm
-> CLS -> text_proj -> L2 normalise. Used to embed search queries; caption_emb.npz was
built with the same encoder.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .onepeace_common import (Layer, Tokenizer, build_and_load, layer_hparams, load_checkpoint,
                              make_token_bucket_position)

LAYERS = 'encoder_wrapper.fusion_model.layers.'
ADAPTER = 'encoder_wrapper.text_adapter.'


class TextAdapter(nn.Module):
    def __init__(self, vocab, dim, heads, bucket_size, n_rel_tables, layernorm_embedding, type_embedding):
        super().__init__()
        self.padding_idx = 1
        self.bucket_size = bucket_size
        self.embed_tokens = nn.Embedding(vocab, dim, padding_idx=1)
        self.embed_positions = nn.Embedding(512 + 2, dim)
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.layernorm_embedding = nn.LayerNorm(dim) if layernorm_embedding else None
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, dim)) if type_embedding else None
        if n_rel_tables:
            self.register_buffer('rp_bucket', self._rp_bucket(bucket_size))
            self.rel_pos_table_list = nn.ModuleList([nn.Embedding(2 * bucket_size + 2, heads)
                                                     for _ in range(n_rel_tables)])
        else:
            self.rel_pos_table_list = None

    @staticmethod
    def _rp_bucket(bucket_size):
        num_rel_dis = 2 * bucket_size - 1
        rp = make_token_bucket_position(bucket_size, max_position=1024)
        rp[0, :] = num_rel_dis
        rp[:, 0] = num_rel_dis + 1
        rp[0, 0] = num_rel_dis + 2
        return rp

    def forward(self, src_tokens):
        bsz, seq_len = src_tokens.size(0), src_tokens.size(1) + 1
        padding_mask = src_tokens.new_zeros((bsz, seq_len)).bool()
        padding_mask[:, 1:] = src_tokens.eq(self.padding_idx)
        pos_embed = self.embed_positions(torch.arange(seq_len, device=src_tokens.device).expand(bsz, -1))
        bias_list = None
        if self.rel_pos_table_list is not None:
            rp = self.rp_bucket[:seq_len, :seq_len]
            bias_list = [t(rp).unsqueeze(0).expand(bsz, -1, -1, -1).permute(0, 3, 1, 2)
                         for t in self.rel_pos_table_list]
        x = torch.cat([self.cls_embedding.expand(bsz, -1, -1), self.embed_tokens(src_tokens)], dim=1)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = x + pos_embed
        if self.type_embedding is not None:
            x = x + self.type_embedding.expand_as(x)
        return x, padding_mask, bias_list


class OnePeaceText(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.heads = hp['heads']
        self.text_adapter = TextAdapter(hp['vocab'], hp['dim'], hp['heads'], hp['bucket_size'],
                                        hp['n_rel_tables'], hp['layernorm_embedding'], hp['type_embedding'])
        self.layers = nn.ModuleList([
            Layer(hp['dim'], hp['ffn_dim'], hp['heads'], hp['scale_attn'], hp['scale_fc'], hp['scale_heads'],
                  hp['magneto'], hp['layer_scale'], 'text_ffn') for _ in range(hp['layers'])])
        self.text_layer_norm = nn.LayerNorm(hp['dim'])
        self.text_proj = nn.Linear(hp['dim'], hp['dim'])

    def init_buffers(self):
        a = self.text_adapter
        if getattr(a, 'rp_bucket', None) is not None and a.rp_bucket.is_meta:
            a.rp_bucket = TextAdapter._rp_bucket(a.bucket_size)

    def forward(self, src_tokens):
        x, pad, bias_list = self.text_adapter(src_tokens)
        B, T, _ = x.shape
        has_pads = bool(pad.any())
        if has_pads:
            x = x * (1 - pad.unsqueeze(-1).type_as(x))
        biases = []
        for b in bias_list or []:
            b = b.clone()
            if has_pads:
                b.masked_fill_(pad.view(B, 1, 1, T).expand(-1, self.heads, T, -1), float('-inf'))
            biases.append(b)
        x = x.transpose(0, 1)
        for i, layer in enumerate(self.layers):
            bias = None if not biases else biases[0] if len(biases) == 1 else biases[i]
            x = layer(x, bias)
        x = self.text_layer_norm(x)
        return F.normalize(self.text_proj(x[0]), dim=1)


def _remap(sd):
    out = {}
    for k, v in sd.items():
        k = k.replace('encoder_wrapper.fusion_model.text_layer_norm.', 'text_layer_norm.')
        k = k.replace(LAYERS, 'layers.').replace(ADAPTER, 'text_adapter.')
        out[k] = v
    return out


class TextEncoder:
    """Sentences -> (N, 1536) normalised ONE-PEACE text vectors."""

    def __init__(self, checkpoint, device, dtype):
        ck = load_checkpoint(checkpoint)
        sd = ck['state_dict'] if 'state_dict' in ck else ck['model']
        sd = {k: v for k, v in sd.items() if k.startswith((ADAPTER, 'encoder_wrapper.fusion_model.text_layer_norm.', 'text_proj.'))
              or (k.startswith(LAYERS) and not k.split('.', 4)[4].startswith(('image_ffn', 'audio_ffn')))}
        hp = layer_hparams(sd, LAYERS, 'text_ffn')
        n_rel = sum(1 for k in sd if k.startswith(ADAPTER + 'rel_pos_table_list.') and k.endswith('.weight'))
        hp.update(vocab=sd[ADAPTER + 'embed_tokens.weight'].shape[0], n_rel_tables=n_rel,
                  heads=sd[ADAPTER + 'rel_pos_table_list.0.weight'].shape[1],
                  bucket_size=(sd[ADAPTER + 'rel_pos_table_list.0.weight'].shape[0] - 2) // 2,
                  layernorm_embedding=ADAPTER + 'layernorm_embedding.weight' in sd,
                  type_embedding=ADAPTER + 'type_embedding' in sd)
        self.tokenizer = Tokenizer()
        self.device, self.dtype = device, dtype
        self.model = build_and_load(lambda: OnePeaceText(hp), _remap(sd), device, dtype)

    @torch.no_grad()
    def __call__(self, texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            out.append(self.model(self.tokenizer(texts[i:i + batch]).to(self.device)).float())
        return torch.cat(out)
