import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (get_sinusoid_encoding, TransformerBlock,
                    MaskedConv1D, LayerNorm)


class TextBridge(nn.Module):
    """
        The text stream (T) at one pyramid level (model.text_stream).

        T is one token per sentence: the query, or a learned null token for a video alone.
          1. T reads the level:     t = t + Attn(q = t, k = v = the V and A steps), then an MLP
          2. V and A each read T:   X = X + tanh(g_X) * Attn(q = X, k = v = [sink, t])
        The learned sink key lets a step ignore the sentence. The gates g_V, g_A start at 0, so at
        init V and A leave the bridge unchanged and the backbone is the two-stream one.
    """
    def __init__(self, n_embd, n_head, proj_pdrop=0.0):
        super().__init__()
        self.ln_t = nn.LayerNorm(n_embd)
        self.ln_x = nn.LayerNorm(n_embd)
        self.read = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Dropout(proj_pdrop),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln_k = nn.LayerNorm(n_embd)
        self.sink = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.write = nn.ModuleDict({m: nn.MultiheadAttention(n_embd, n_head, batch_first=True) for m in 'VA'})
        self.ln_w = nn.ModuleDict({m: nn.LayerNorm(n_embd) for m in 'VA'})
        self.gate = nn.ParameterDict({m: nn.Parameter(torch.zeros(1)) for m in 'VA'})

    def forward(self, x_V, x_A, mask, t):
        # x_V, x_A (Q, C, T) one row per (video, sentence) pair, mask (Q, 1, T), t (Q, 1, C)
        v, a = x_V.transpose(1, 2), x_A.transpose(1, 2)                  # (Q, T, C)
        steps = self.ln_x(torch.cat((v, a), dim=1))                      # (Q, 2T, C)
        pad = ~mask.squeeze(1).repeat(1, 2)                              # (Q, 2T), True = padding
        t = t + self.read(self.ln_t(t), steps, steps, key_padding_mask=pad, need_weights=False)[0]
        t = t + self.mlp(t)
        keys = torch.cat((self.sink.expand(len(t), -1, -1), self.ln_k(t)), dim=1)   # (Q, 2, C)
        valid = mask.transpose(1, 2).to(x_V.dtype)                       # (Q, T, 1)
        out = []
        for m, x in (('V', v), ('A', a)):
            w = self.write[m](self.ln_w[m](x), keys, keys, need_weights=False)[0]
            out.append((x + torch.tanh(self.gate[m]) * w * valid).transpose(1, 2))
        return out[0], out[1], t


class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers, one stream per modality (from UniAV).

        Visual (V) and audio (A) features each go through: conv embedding -> arch[1] - 1 self-attention
        blocks -> one cross-modal block at full resolution -> arch[2] cross-modal blocks that each halve
        the length. Returns the arch[2] + 1 pyramid levels of each stream, and their masks.

        Cross-modal block of stream X with the other stream Y (TransformerBlock(x1=X, x2=Y)):
        out = X + Attention(query = Y, key = X, value = X), i.e. X is re-weighted over time by
        what Y finds relevant. With pyramid_attn 'self', Y = X.

        With text_dim > 0 a sentence is the third stream (TextBridge after level 0 and after every
        pyramid level). The stem does not see it; from level 0 on, every (video, sentence) pair is
        its own row, so the outputs have one row per sentence. No sentence: a learned null token.
    """
    def __init__(
        self,
        n_in_V,                # input visual feature dimension
        n_in_A,                # input audio feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length (int)
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        scale_factor = 2,      # dowsampling rate for the branch,
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        pyramid_attn = 'cross',  # 'cross': level-0 and pyramid blocks attend from one stream to the other;
                                 # 'self': each stream attends to itself (intra-modal only, OV-AVEL Table A9)
        text_dim = 0,          # > 0: sentence vectors of this size are a third stream (TextBridge)
    ):
        super().__init__()
        assert len(arch) == 3
        assert pyramid_attn in ('cross', 'self'), pyramid_attn
        self.cross = pyramid_attn == 'cross'
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        # 768 -> 512
        self.embd_V = nn.ModuleList()
        self.embd_A = nn.ModuleList()
        self.embd_norm_V = nn.ModuleList()
        self.embd_norm_A = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels_V = n_in_V
                in_channels_A = n_in_A
            else:
                in_channels_V = n_embd
                in_channels_A = n_embd
            self.embd_V.append(MaskedConv1D(
                    in_channels_V, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            self.embd_A.append(MaskedConv1D(
                    in_channels_A, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm_V.append(
                    LayerNorm(n_embd)
                )
                self.embd_norm_A.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm_V.append(nn.Identity())
                self.embd_norm_A.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.self_att_V = nn.ModuleList()
        self.self_att_A = nn.ModuleList()

        for idx in range(arch[1]-1): 
            self.self_att_V.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.self_att_A.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
        #cross-attention on original temporal resolution
        self.ori_cross_att_Va = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
        self.ori_cross_att_Av = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )

        #cross-attention after down-sampling
        self.cross_att_Va = nn.ModuleList()
        self.cross_att_Av = nn.ModuleList()
        for idx in range(arch[2]):
            self.cross_att_Va.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.cross_att_Av.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        # text stream: sentence -> one token, a null token for a video alone, one bridge per level
        self.bridges = None
        if text_dim > 0:
            self.text_embd = nn.Sequential(nn.Linear(text_dim, n_embd), nn.LayerNorm(n_embd))
            self.text_null = nn.Parameter(torch.randn(1, 1, n_embd) * 0.02)
            self.bridges = nn.ModuleList(TextBridge(n_embd, n_head, proj_pdrop) for _ in range(arch[2] + 1))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x_V, x_A, mask, text=None, vid=None):
        # x_V/x_A: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        # text stream only: text (Q, text_dim) sentence vectors, vid (Q,) the video of each sentence;
        # text None: one null token per video
        B, _, T = x_V.size()
        mask_V = mask_A = mask
        # embedding network
        for idx in range(len(self.embd_V)):
            x_V, mask_V = self.embd_V[idx](x_V, mask_V) 
            x_V = self.relu(self.embd_norm_V[idx](x_V))

            x_A, mask_A = self.embd_A[idx](x_A, mask_A)
            x_A = self.relu(self.embd_norm_A[idx](x_A))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        # stem transformer
        for idx in range(len(self.self_att_V)):
            x_V, mask_V = self.self_att_V[idx](x_V, x_V, mask_V)
            x_A, mask_A = self.self_att_A[idx](x_A, x_A, mask_A)

        # text stream: from here on one row per (video, sentence) pair
        if self.bridges is not None:
            if text is None:
                t = self.text_null.expand(B, -1, -1)                   # (B, 1, C)
            else:
                x_V, x_A, mask_V, mask_A = x_V[vid], x_A[vid], mask_V[vid], mask_A[vid]
                t = self.text_embd(text)[:, None]                      # (Q, 1, C)

        # level 0: x_Va = V updated with audio queries, x_Av = A updated with visual queries (see the
        # class docstring); the same weights either way: with pyramid_attn 'self' the queries come
        # from the stream itself
        x_Va, mask_V = self.ori_cross_att_Va(x_V, x_A if self.cross else x_V, mask_V)
        x_Av, mask_A = self.ori_cross_att_Av(x_A, x_V if self.cross else x_A, mask_A)
        if self.bridges is not None:
            x_Va, x_Av, t = self.bridges[0](x_Va, x_Av, mask_V, t)

        # prep for outputs
        out_feats_V = tuple()
        out_feats_A = tuple()
        out_masks_V = tuple()
        # 1x resolution
        out_feats_V += (x_Va, )
        out_masks_V += (mask_V, )
        out_feats_A += (x_Av, )

        # main branch with downsampling: level idx + 1 is built from level idx of both streams
        for idx in range(len(self.cross_att_Va)):
            x_V, mask_V = self.cross_att_Va[idx](out_feats_V[idx], out_feats_A[idx] if self.cross else out_feats_V[idx], mask_V)
            x_A, mask_A = self.cross_att_Av[idx](out_feats_A[idx], out_feats_V[idx] if self.cross else out_feats_A[idx], mask_A)
            if self.bridges is not None:
                x_V, x_A, t = self.bridges[idx + 1](x_V, x_A, mask_V, t)

            out_feats_V += (x_V, )
            out_masks_V += (mask_V, )
            out_feats_A += (x_A, )

        return out_feats_V, out_feats_A, out_masks_V