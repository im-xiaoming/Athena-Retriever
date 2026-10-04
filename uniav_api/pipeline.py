"""Video -> events with captions and retrieval vectors, and text search over processed videos.

    from uniav_api.pipeline import UniAVPipeline
    pipe = UniAVPipeline()
    result = pipe.process('cooking.mp4')           # events: start, end, score, caption, embedding
    result = pipe.process_sample('6uHoTJSLoL8')    # the same from features shipped in uniav_api/samples
    hits = pipe.search('cut the onion', top_k=5)   # events of processed videos matching a query

Event vectors and query vectors live in the same 512-d caption space (the space the model
projects ONE-PEACE caption vectors into with clip_proj), so search is a cosine.

The checkpoint decides which features the model takes (features.py): an API checkpoint made by
tools/export_api_ckpt.py carries its training config; a raw training checkpoint uses
Config.model_config instead.
"""
import json
import os
import time

import numpy as np
import torch
import yaml

from .captioner import Captioner, PoolQuery
from .config import Config, keep_encoders, pick_device
from .features import FeatureSpec
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
        if not os.path.exists(self.cfg.checkpoint):
            raise FileNotFoundError('checkpoint %s not found; see uniav_api/README.md, "Files needed"'
                                    % self.cfg.checkpoint)
        ck = _load_ckpt(self.cfg.checkpoint)
        if 'config' in ck:   # API checkpoint: its own training config
            mcfg = ck['config']
        else:                # raw training checkpoint: the yaml config
            with open(self.cfg.model_config) as f:
                mcfg = yaml.safe_load(f)
        self.run = ck.get('run', os.path.basename(os.path.dirname(self.cfg.checkpoint)))
        self.spec = FeatureSpec(mcfg['dataset'])
        self.max_seq_len = self.spec.max_seq_len
        sd = {k: v.float() for k, v in ck['state_dict'].items()}
        omni_dim = sd['omni_proj.2.weight'].shape[0] if 'omni_proj.2.weight' in sd else 0
        model_cfg = dict(mcfg['model'], input_dim_V=self.spec.dims[0], input_dim_A=self.spec.dims[1],
                         omni_dim=omni_dim)
        self.model = EventCaptionModel(**model_cfg)
        self.model.load_state_dict(sd, strict=True)
        self.model = self.model.to(self.device).eval()   # small (138M): fp32 everywhere
        self.captioner = Captioner(self.cfg.caption_pool, self.model, self.device)
        self._encoder = self._text = self._pool_query = None
        # one index per model: event vectors of different checkpoints are not comparable
        self.index_dir = os.path.join(self.cfg.index_dir, self.run)
        self.index = {}
        os.makedirs(self.index_dir, exist_ok=True)
        for f in os.listdir(self.index_dir):
            if f.endswith('.json'):
                with open(os.path.join(self.index_dir, f)) as fh:
                    r = json.load(fh)
                self.index[r['video_id']] = r

    # ------------------------------------------------------------------ encoders (lazy)
    @property
    def encoder(self):
        if self._encoder is None:
            keep = keep_encoders(self.cfg, self.device)
            if self.spec.source == 'iv2':
                from .encoders.internvideo2 import InternVideo2AVEncoder
                for p in (self.cfg.iv2_video_encoder, self.cfg.iv2_audio_encoder, self.cfg.iv2_repo):
                    if not os.path.exists(p):
                        raise FileNotFoundError('%s not found: describe_video needs the InternVideo2 encoders, '
                                                'see uniav_api/README.md' % p)
                self._encoder = InternVideo2AVEncoder(self.cfg.iv2_video_encoder, self.cfg.iv2_audio_encoder,
                                                      self.cfg.iv2_repo, self.device, self.spec.video_keys,
                                                      self.spec.l2norm, keep_loaded=keep)
            else:
                from .encoders.onepeace import OnePeaceAVEncoder
                self._encoder = OnePeaceAVEncoder(self.cfg.video_encoder, self.cfg.audio_encoder, self.device,
                                                  self.dtype, keep_loaded=keep)
        return self._encoder

    @property
    def text_encoder(self):
        """ONE-PEACE text encoder, or None when its 6 GB checkpoint is not available."""
        if self._text is None and os.path.exists(self.cfg.text_encoder):
            from .encoders.onepeace_text import TextEncoder
            self._text = TextEncoder(self.cfg.text_encoder, self.device, self.dtype)
        return self._text

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def process_features(self, visual, audio, duration):
        """Features (T, C) in the checkpoint's format (see features.py) -> events, times in seconds."""
        fv, fa, n = self.spec.prepare(visual, audio)
        segs, scores, vecs = self.model(fv.to(self.device), fa.to(self.device))
        secs = self.spec.to_seconds(segs, n, duration)
        keep = select_events(secs, scores, self.cfg.min_score, self.cfg.max_overlap, self.cfg.max_events)
        caps = self.captioner.caption(vecs[keep], alternatives=self.cfg.alternatives) if keep else []
        events = []
        for c, i in zip(caps, keep):
            events.append(dict(start=round(float(secs[i][0]), 2), end=round(float(secs[i][1]), 2),
                               score=round(float(scores[i]), 4), **c,
                               embedding=vecs[i].float().cpu().numpy()))
        return events

    def process_stored(self, visual, audio, duration, video_id, store=True, has_audio=True):
        """Like process(), but from precomputed features."""
        t0 = time.time()
        events = self.process_features(visual, audio, duration)
        result = {'video_id': video_id, 'duration': round(float(duration), 2), 'has_audio': has_audio,
                  'num_events': len(events), 'timing': {'encode_s': 0.0, 'model_s': round(time.time() - t0, 2)},
                  'device': str(self.device), 'model': self.run, 'events': events}
        if store:
            self._store(result)
        return result

    def samples(self):
        """Ids of the videos with features shipped in Config.samples_dir."""
        d = self.cfg.samples_dir
        return sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npz')) if os.path.isdir(d) else []

    def process_sample(self, video_id, store=True):
        """A video from Config.samples_dir (InternVideo2 features + duration) -> events."""
        if self.spec.source != 'iv2':
            raise ValueError('the samples hold InternVideo2 features; checkpoint %s takes %s'
                             % (self.run, self.spec.source))
        z = np.load(os.path.join(self.cfg.samples_dir, video_id + '.npz'))
        visual, audio = self.spec.from_npz(z)
        return self.process_stored(visual, audio, float(z['duration']), video_id, store=store)

    def _store(self, result):
        self.index[result['video_id']] = to_json(result)
        with open(os.path.join(self.index_dir, result['video_id'] + '.json'), 'w') as f:
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
                  'device': str(self.device), 'model': self.run, 'events': events}
        if store:
            self._store(result)
        return result

    def _cached_features(self, path, video_id):
        """Encoder features, reused from cfg.feature_cache when the same file was seen before."""
        if not self.cfg.feature_cache:
            return self.encoder.encode(path)
        st = os.stat(path)
        tag = '' if self.spec.source == 'onepeace' else '_' + self.spec.source + '_' + '_'.join(self.spec.video_keys)
        f = os.path.join(self.cfg.feature_cache, '%s_%d_%d%s.npz' % (video_id, st.st_size, int(st.st_mtime), tag))
        if os.path.exists(f):
            z = np.load(f)
            return {'visual': z['visual'], 'audio': z['audio'], 'duration': float(z['duration']),
                    'has_audio': bool(z['has_audio'])}
        feats = self.encoder.encode(path)
        os.makedirs(self.cfg.feature_cache, exist_ok=True)
        np.savez(f, **feats)
        return feats

    # ------------------------------------------------------------------ search
    @torch.no_grad()
    def embed_query(self, text):
        """Query -> 512-d vector in the event space.

        With the ONE-PEACE text encoder: its vector of the query through the model's clip_proj.
        Without it (the encoder is 6 GB): the train captions closest to the query by word overlap
        (TF-IDF), averaged in the event space; good for queries phrased like recipe steps.
        """
        enc = self.text_encoder
        if enc is not None:
            raw = enc([text]).to(self.device)
            return self.model.embed_head.embed_captions(raw.float())[0]
        if self._pool_query is None:
            print('note: ONE-PEACE text encoder not found (%s); search matches queries through the '
                  'train captions instead' % self.cfg.text_encoder, flush=True)
            self._pool_query = PoolQuery(self.captioner)
        return self._pool_query(text)

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
    """numpy vectors -> lists, for the on-disk index."""
    out = dict(result)
    out['events'] = [dict(e, embedding=[round(float(x), 5) for x in e['embedding']]) for e in result['events']]
    return out
