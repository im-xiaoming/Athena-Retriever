"""Caption generation for events: GPT-2 conditioned on the event (tools/capgen/train_capgen.py).

The event model gives each event a vector q and K level-0 feature tokens; a small MLP maps them to
GPT-2 input embeddings (a prefix), optionally followed by the best retrieved train captions as text
(RAG checkpoints), then " caption:" and GPT-2 writes the caption. On YouCook2 validation spans this
beats the retrieval pick: CIDEr 97.6 vs 88.9 (see experiments/NOTES.md).

Checkpoint: tools/capgen/export_generator.py -> ckpt/api/capgen_<name>.pth (fp16 GPT-2 + prefix).
GPT-2's tokenizer and config come from HuggingFace (openai-community/gpt2, public).
"""
import torch
import torch.nn as nn


class Prefix(nn.Module):
    """q and the K feature tokens -> K + 1 GPT-2 input embeddings (same as in train_capgen.py)."""

    def __init__(self, d_in, d_model, n_tok):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.mlp = nn.Sequential(nn.Linear(d_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos = nn.Parameter(torch.zeros(n_tok, d_model))

    def forward(self, q, tok):
        x = torch.cat((q[:, None], tok), 1)
        return self.mlp(self.norm(x)) + self.pos


class CaptionGenerator:
    def __init__(self, checkpoint, device):
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        args = ck['args']
        self.rag, self.beams, self.device = args.get('rag', 0), args.get('beams', 3), device
        self.tokenizer = GPT2TokenizerFast.from_pretrained(args['gpt'])
        self.gpt = GPT2LMHeadModel(GPT2Config.from_pretrained(args['gpt']))
        self.gpt.load_state_dict({k: v.float() for k, v in ck['gpt'].items()}, strict=False)   # tied lm_head
        pre = {k: v.float() for k, v in ck['prefix'].items()}
        self.prefix = Prefix(pre['norm.weight'].shape[0], self.gpt.config.n_embd, pre['pos'].shape[0])
        self.prefix.load_state_dict(pre)
        self.gpt.to(device).eval(); self.prefix.to(device).eval()
        self.k = pre['pos'].shape[0] - 1

    def _text(self, cands):
        return (' candidates: ' + ' ; '.join(cands[:self.rag]) + ' .' if self.rag else '') + ' caption:'

    @torch.no_grad()
    def __call__(self, q, tok, candidates=None, batch=32):
        """q (N, D), tok (N, K, D), candidates: N lists of retrieved captions (RAG only) -> N captions."""
        out, pad = [], self.tokenizer.eos_token_id
        for s in range(0, len(q), batch):
            qs, ts = q[s:s + batch].float().to(self.device), tok[s:s + batch].float().to(self.device)
            pre = self.prefix(qs, ts)                                                   # (B, P, D)
            texts = [self.tokenizer.encode(self._text(candidates[s + i] if candidates else [])) for i in range(len(qs))]
            L = max(len(t) for t in texts); P = pre.shape[1]
            ids = torch.full((len(qs), L), pad, dtype=torch.long)
            for i, t in enumerate(texts):
                ids[i, L - len(t):] = torch.tensor(t)
            emb = self.gpt.transformer.wte(ids.to(self.device))
            # left padding goes before the prefix: [pad, prefix, text], exactly as in training
            rows, masks = [], []
            for i, t in enumerate(texts):
                n_pad = L - len(t)
                rows.append(torch.cat((emb[i, :n_pad], pre[i], emb[i, n_pad:])))
                masks.append(torch.cat((torch.zeros(n_pad, dtype=torch.long), torch.ones(P + len(t), dtype=torch.long))))
            gen = self.gpt.generate(inputs_embeds=torch.stack(rows), attention_mask=torch.stack(masks).to(self.device),
                                    max_new_tokens=30, num_beams=self.beams, no_repeat_ngram_size=3,
                                    pad_token_id=pad, eos_token_id=pad)
            out += [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]
        return out
