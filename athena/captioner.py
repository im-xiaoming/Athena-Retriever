"""Caption pool and caption choice for event vectors.

The pool holds the unique train captions and their InternVideo2 text vectors (the model never
saw validation captions; assets/caption_pool.npz, with the centring mean used for new sentences).
Captions are projected once into the event space with the model's clip_proj: their InternVideo2
vectors, or, for a model trained in the teacher's text space, the vectors stored in the API
checkpoint (caption_pool). An event vector is matched against them by cosine and the caption is
picked by consensus (minimum Bayes risk over the top 20) in InternVideo2 text space, exactly as
train.py scores it.
"""
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F


def build_caption_pool(caption_emb, annotations, out):
    """caption_emb_iv2.npz + annotations -> unique train captions with their InternVideo2 vectors."""
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
    np.savez_compressed(out, texts=np.array(texts, dtype=str), emb=np.stack(embs).astype(np.float16),
                        mean=np.asarray(z['mean'], np.float32))
    return len(texts)


class Captioner:
    """Picks a caption for each event vector from the train caption pool.

    Two views of the pool, in the same order as texts:
      pool_f  captions in the event space (the model's clip_proj): what an event vector is scored against
      pool_n  normalised InternVideo2 text vectors: where candidates are compared with each other
    pool_train (from the API checkpoint) replaces the vectors of assets/caption_pool.npz when the model
    was trained on other caption vectors (another centring, or the teacher's text space).
    """

    def __init__(self, pool_path, model, device, pool_train=None):
        z = np.load(pool_path)
        self.texts = [str(t) for t in z['texts']]
        self.mean = torch.from_numpy(z['mean'].astype(np.float32))   # centring of new sentences
        raw = torch.from_numpy(z['emb'].astype(np.float32)).to(device)
        train = raw
        if pool_train is not None:   # caption vectors of the model's own caption space, same order
            assert len(pool_train['emb']) == len(self.texts), 'caption_pool does not match %s' % pool_path
            train = torch.as_tensor(np.asarray(pool_train['emb'], np.float32), device=device)
            if pool_train.get('mean') is not None:   # InternVideo2 text, centred differently (e.g. *_iv2j.npz)
                raw = train
                self.mean = torch.as_tensor(np.asarray(pool_train['mean'], np.float32))
        with torch.no_grad():
            self.pool_f = model.embed_head.embed_captions(train.to(next(model.parameters()).dtype)).float()
            self.pool_n = F.normalize(raw, dim=-1)
            self.scale = float(model.embed_head.logit_scale.exp().clamp(max=100))

    @torch.no_grad()
    def caption(self, vecs, k=20, alternatives=3):
        """vecs (N, D) normalised -> list of {caption, similarity, consensus, alternatives}.

        Minimum Bayes risk pick, as train.py's mbr_pick: the k best captions by cosine are weighted by
        softmax(scale * cosine), and the one most similar to the weighted group wins (consensus).
        similarity is that caption's cosine with the event; alternatives are the next best by cosine.
        """
        out = []
        for q in vecs.float():
            s = self.pool_f @ q                           # cosine with every pool caption
            top = s.topk(k)
            p = F.softmax(self.scale * top.values, dim=0)   # weight of each candidate
            cand = self.pool_n[top.indices]
            agree = (cand @ cand.t()) @ p                 # weighted agreement of each candidate with the others
            order = agree.argsort(descending=True)
            best = int(top.indices[order[0]])
            alts = [{'caption': self.texts[int(top.indices[j])], 'similarity': round(float(top.values[j]), 4)}
                    for j in range(min(alternatives + 1, k)) if int(top.indices[j]) != best][:alternatives]
            out.append({'caption': self.texts[best], 'similarity': round(float(s[best]), 4),
                        'consensus': round(float(agree[order[0]]), 4), 'alternatives': alts})
        return out


class PoolQuery:
    """Text query -> event-space vector without a text encoder.

    Ranks the train captions by TF-IDF cosine with the query and averages the projected vectors
    of the best ones, weighted by that similarity. Works for queries phrased like recipe steps
    ("cut the onion"); words never used in a train caption are ignored.
    """

    def __init__(self, captioner, top=5):
        self.top, self.cap = top, captioner
        docs = [self._words(t) for t in captioner.texts]
        vocab = sorted({w for d in docs for w in d})
        self.vid = {w: i for i, w in enumerate(vocab)}   # word -> column
        df = np.zeros(len(vocab), np.float32)            # document frequency of every word
        for d in docs:
            for w in set(d):
                df[self.vid[w]] += 1
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1   # smoothed idf, as scikit-learn
        rows = np.zeros((len(docs), len(vocab)), np.float32)
        for i, d in enumerate(docs):
            for w in d:
                rows[i, self.vid[w]] += 1
        rows *= self.idf
        self.docs = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)   # (N captions, V)

    @staticmethod
    def _words(text):
        return re.findall(r"[a-z]+", text.lower())

    def __call__(self, text):
        v = np.zeros(len(self.vid), np.float32)
        for w in self._words(text):
            if w in self.vid:
                v[self.vid[w]] += 1
        v *= self.idf
        if not v.any():
            raise ValueError('none of the words in %r appear in the train captions' % text)
        sim = self.docs @ (v / np.linalg.norm(v))
        best = np.argsort(-sim)[:self.top]
        w = torch.from_numpy(sim[best]).to(self.cap.pool_f.device)
        q = (w[:, None] * self.cap.pool_f[torch.from_numpy(best).to(self.cap.pool_f.device)]).sum(0)
        return F.normalize(q, dim=-1)
