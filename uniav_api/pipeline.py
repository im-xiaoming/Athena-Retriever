"""Video -> events with captions and retrieval vectors, and text search over processed videos.

    from uniav_api.pipeline import UniAVPipeline
    pipe = UniAVPipeline()
    result = pipe.process('cooking.mp4')           # events: start, end, score, caption, embedding
    hits = pipe.search('cut the onion', top_k=5)   # events of processed videos matching a query

Event vectors and query vectors live in the same 512-d caption space (the space the model
projects ONE-PEACE caption vectors into with clip_proj), so search is a cosine.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .captioner import Captioner
from .config import Config, keep_encoders, pick_device
from .model.event_model import EventCaptionModel


def _load_ckpt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def select_events(segs, scores, min_score, max_overlap, max_events):
    """Best-first greedy pick: score >= min_score, IoU with every kept event <= max_overlap."""
    kept = []
    for i in np.argsort(-scores, kind='stable'):
        if scores[i] < min_score or len(kept) >= max_events:
            break
        a = segs[i]
        ok = True
        for j in kept:
            b = segs[j]
            inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
            union = (a[1] - a[0]) + (b[1] - b[0]) - inter
            if union > 0 and inter / union > max_overlap:
                ok = False; break
        if ok:
            kept.append(i)
    return sorted(kept, key=lambda i: segs[i][0])


class UniAVPipeline:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config.from_env()
        self.device, self.dtype = pick_device(self.cfg.device)
        with open(self.cfg.model_config) as f:
            mcfg = yaml.safe_load(f)
        self.max_seq_len = mcfg['dataset']['max_seq_len']
        self.grid = mcfg['dataset']   # feat_stride / num_frames / default_fps of the training features
        sd = _load_ckpt(self.cfg.checkpoint)['state_dict']
        omni_dim = sd['omni_proj.2.weight'].shape[0] if 'omni_proj.2.weight' in sd else 0
        self.model = EventCaptionModel(**mcfg['model'], omni_dim=omni_dim)
        self.model.load_state_dict(sd, strict=True)
        self.model = self.model.to(self.device).eval()   # small (138M): fp32 everywhere
        self.captioner = Captioner(self.cfg.caption_pool, self.model, self.device)
        self._encoder = self._text = None
        self.index = {}
        os.makedirs(self.cfg.index_dir, exist_ok=True)
        for f in os.listdir(self.cfg.index_dir):
            if f.endswith('.json'):
                with open(os.path.join(self.cfg.index_dir, f)) as fh:
                    r = json.load(fh)
                self.index[r['video_id']] = r

    # ------------------------------------------------------------------ encoders (lazy)
    @property
    def encoder(self):
        if self._encoder is None:
            from .encoders.onepeace import OnePeaceAVEncoder
            self._encoder = OnePeaceAVEncoder(self.cfg.video_encoder, self.cfg.audio_encoder, self.device,
                                              self.dtype, keep_loaded=keep_encoders(self.cfg, self.device))
        return self._encoder

    @property
    def text_encoder(self):
        if self._text is None:
            from .encoders.onepeace_text import TextEncoder
            self._text = TextEncoder(self.cfg.text_encoder, self.device, self.dtype)
        return self._text

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def process_features(self, visual, audio, duration):
        """Features (T, 1536) in the training format -> events (start/end in seconds)."""
        n = min(len(visual), len(audio))
        fv = torch.from_numpy(np.ascontiguousarray(visual[:n].T, dtype=np.float32))
        fa = torch.from_numpy(np.ascontiguousarray(audio[:n].T, dtype=np.float32))
        if n != self.max_seq_len:   # force_upsampling: every video becomes max_seq_len steps
            fv = F.interpolate(fv[None], size=self.max_seq_len, mode='linear', align_corners=False)[0]
            fa = F.interpolate(fa[None], size=self.max_seq_len, mode='linear', align_corners=False)[0]
        segs, scores, vecs = self.model(fv.to(self.device), fa.to(self.device))
        # grid -> seconds, exactly as the dataset does
        stride = float((n - 1) * self.grid['feat_stride'] + self.grid['num_frames']) / self.max_seq_len
        fps = self.grid['default_fps']
        secs = np.clip((segs * stride + 0.5 * stride) / fps, 0.0, float(duration))
        keep = select_events(secs, scores, self.cfg.min_score, self.cfg.max_overlap, self.cfg.max_events)
        caps = self.captioner.caption(vecs[keep], alternatives=self.cfg.alternatives) if keep else []
        events = []
        for c, i in zip(caps, keep):
            events.append(dict(start=round(float(secs[i][0]), 2), end=round(float(secs[i][1]), 2),
                               score=round(float(scores[i]), 4), **c,
                               embedding=vecs[i].float().cpu().numpy()))
        return events

    def process_stored(self, visual, audio, duration, video_id, store=True):
        """Like process(), but from precomputed features (e.g. data/youcookii/av_features)."""
        t0 = time.time()
        events = self.process_features(visual, audio, duration)
        result = {'video_id': video_id, 'duration': round(float(duration), 2), 'has_audio': True,
                  'num_events': len(events), 'timing': {'encode_s': 0.0, 'model_s': round(time.time() - t0, 2)},
                  'device': str(self.device), 'events': events}
        if store:
            self._store(result)
        return result

    def _store(self, result):
        self.index[result['video_id']] = to_json(result)
        with open(os.path.join(self.cfg.index_dir, result['video_id'] + '.json'), 'w') as f:
            json.dump(self.index[result['video_id']], f)

    def process(self, path, video_id=None, store=True):
        t0 = time.time()
        video_id = video_id or os.path.splitext(os.path.basename(path))[0]
        feats = self._cached_features(path, video_id)
        t1 = time.time()
        events = self.process_features(feats['visual'], feats['audio'], feats['duration'])
        result = {'video_id': video_id, 'duration': round(float(feats['duration']), 2),
                  'has_audio': bool(feats['has_audio']), 'num_events': len(events),
                  'timing': {'encode_s': round(t1 - t0, 1), 'model_s': round(time.time() - t1, 2)},
                  'device': str(self.device), 'events': events}
        if store:
            self._store(result)
        return result

    def _cached_features(self, path, video_id):
        """Encoder features, reused from cfg.feature_cache when the same file was seen before."""
        if not self.cfg.feature_cache:
            return self.encoder.encode(path)
        st = os.stat(path)
        f = os.path.join(self.cfg.feature_cache, '%s_%d_%d.npz' % (video_id, st.st_size, int(st.st_mtime)))
        if os.path.exists(f):
            z = np.load(f)
            return {'visual': z['visual'], 'audio': z['audio'], 'duration': float(z['duration']),
                    'has_audio': bool(z['has_audio'])}
        feats = self.encoder.encode(path)
        os.makedirs(self.cfg.feature_cache, exist_ok=True)
        np.savez(f, **feats)
        return feats

    @torch.no_grad()
    def embed_query(self, text):
        """Query -> 512-d vector in the event space (ONE-PEACE text -> the model's clip_proj)."""
        raw = self.text_encoder([text]).to(self.device)
        return self.model.embed_head.embed_captions(raw.float())[0]

    def search(self, query, top_k=10, video_ids=None):
        q = self.embed_query(query).cpu().numpy()
        hits = []
        for vid, r in self.index.items():
            if video_ids and vid not in video_ids:
                continue
            for e in r['events']:
                hits.append({'video_id': vid, 'start': e['start'], 'end': e['end'], 'caption': e['caption'],
                             'score': round(float(np.dot(q, np.asarray(e['embedding'], np.float32))), 4)})
        hits.sort(key=lambda h: -h['score'])
        return hits[:top_k]


def to_json(result):
    """numpy vectors -> lists, for HTTP responses and the on-disk index."""
    out = dict(result)
    out['events'] = [dict(e, embedding=[round(float(x), 5) for x in e['embedding']]) for e in result['events']]
    return out
