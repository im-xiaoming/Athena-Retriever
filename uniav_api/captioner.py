"""Caption pool and caption choice for event vectors.

The pool holds the unique train captions and their ONE-PEACE text vectors (the model never
saw validation captions). Captions are projected once into the model's caption space with
its clip_proj; an event vector is matched against them by cosine and the caption is picked
by consensus (minimum Bayes risk over the top 20), as in train_event.py.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def build_caption_pool(caption_emb, annotations, out):
    """caption_emb.npz + annotations -> unique train captions with their ONE-PEACE vectors."""
    z = np.load(caption_emb, allow_pickle=True)
    with open(annotations) as f:
        subset = {v: x['subset'] for v, x in json.load(f)['database'].items()}
    texts, embs, seen = [], [], set()
    for key, text, emb in zip(z['keys'], z['sentences'], z['emb']):
        text = str(text)
        if subset[str(key).rsplit('#', 1)[0]] != 'training' or text in seen:
            continue
        seen.add(text); texts.append(text); embs.append(emb)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # plain unicode array, no pickle: readable by any numpy version (1.x and 2.x)
    np.savez_compressed(out, texts=np.array(texts, dtype=str), emb=np.stack(embs).astype(np.float16))
    return len(texts)


class Captioner:
    def __init__(self, pool_path, model, device):
        z = np.load(pool_path)
        self.texts = [str(t) for t in z['texts']]
        raw = torch.from_numpy(z['emb'].astype(np.float32)).to(device)
        with torch.no_grad():
            self.pool_f = model.embed_head.embed_captions(raw.to(next(model.parameters()).dtype)).float()
            self.pool_n = F.normalize(raw, dim=-1)
            self.scale = float(model.embed_head.logit_scale.exp().clamp(max=100))

    @torch.no_grad()
    def caption(self, vecs, k=20, alternatives=3):
        """vecs (N, D) normalised -> list of {caption, confidence, alternatives}."""
        out = []
        for q in vecs.float():
            s = self.pool_f @ q
            top = s.topk(k)
            p = F.softmax(self.scale * top.values, dim=0)
            cand = self.pool_n[top.indices]
            agree = (cand @ cand.t()) @ p
            order = agree.argsort(descending=True)
            best = int(top.indices[order[0]])
            alts = [{'caption': self.texts[int(top.indices[j])], 'similarity': round(float(top.values[j]), 4)}
                    for j in range(min(alternatives + 1, k)) if int(top.indices[j]) != best][:alternatives]
            out.append({'caption': self.texts[best], 'similarity': round(float(s[best]), 4),
                        'consensus': round(float(agree[order[0]]), 4), 'alternatives': alts})
        return out
