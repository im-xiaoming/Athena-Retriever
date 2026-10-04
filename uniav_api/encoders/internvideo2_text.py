"""InternVideo2-1B text encoder: sentences -> 512-d vectors in the space of the v512 video features.

InternVideo2 stage 2 encodes text with BERT-large in "text" mode, which runs only the first
fusion_layer = 19 layers (the later ones hold the video cross-attention), takes the [CLS] token
and maps it with text_proj (1024 -> 512). vision_proj maps the pooled video feature v768 to the
same 512-d space (v512 in tools/extract_internvideo2.py). Layers 0..18 are plain BERT layers, so
transformers' BertModel with 19 layers loads them as they are; no upstream code is needed.

Weights come from the same checkpoint as the video encoder (InternVideo2-stage2_1b-224p-f4.pt),
the tokenizer from HuggingFace (google-bert/bert-large-uncased, public).
"""
import torch
import torch.nn.functional as F

FUSION_LAYER = 19
TOKENIZER = 'google-bert/bert-large-uncased'


class InternVideo2TextEncoder:
    dim = 512

    def __init__(self, checkpoint, device, dtype=torch.float16, max_len=40):
        from transformers import BertConfig, BertModel, BertTokenizerFast
        self.device, self.max_len = device, max_len
        self.tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER)
        cfg = BertConfig(vocab_size=30522, hidden_size=1024, num_hidden_layers=FUSION_LAYER,
                         num_attention_heads=16, intermediate_size=4096, max_position_embeddings=512)
        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.proj = torch.nn.Linear(1024, self.dim)
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        sd = ck.get('module', ck.get('model', ck))
        bert = {k[len('text_encoder.bert.'):]: v for k, v in sd.items() if k.startswith('text_encoder.bert.')}
        bert = {k: v for k, v in bert.items()
                if not k.startswith('encoder.layer.') or int(k.split('.')[2]) < FUSION_LAYER}
        missing, unexpected = self.bert.load_state_dict(bert, strict=False)
        missing = [k for k in missing if not k.endswith('position_ids')]   # a buffer, not a weight
        unexpected = [k for k in unexpected if not k.endswith('position_ids')]
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        self.proj.load_state_dict({k[len('text_proj.'):]: v for k, v in sd.items() if k.startswith('text_proj.')})
        del ck, sd
        self.bert.to(device, dtype).eval()
        self.proj.to(device, dtype).eval()

    @torch.no_grad()
    def __call__(self, texts):
        """N sentences -> (N, 512) L2-normalised float32 vectors."""
        t = self.tokenizer([s.strip().lower() for s in texts], padding=True, truncation=True,
                           max_length=self.max_len, return_tensors='pt').to(self.device)
        cls = self.bert(input_ids=t.input_ids, attention_mask=t.attention_mask).last_hidden_state[:, 0]
        return F.normalize(self.proj(cls).float(), dim=-1)
