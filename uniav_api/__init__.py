"""UniAV event captioning as Python functions.

    import uniav_api as uv

    result = uv.describe_video('cooking.mp4')          # events with captions and embeddings
    for e in result['events']:
        print(e['start'], e['end'], e['caption'])

    uv.show(result)                                    # predicted events next to the YouCook2 GT
    hits = uv.search('cut the onion', top_k=5)         # over every video described so far
    vec = uv.embed_text('add salt to the pot')         # 512-d query vector, same space as events

The model and encoders load on first use (GPU if available: CUDA or Apple MPS; else CPU).
Use uv.load(...) to pick a device or checkpoint explicitly, or UniAVPipeline for full control.
"""
from .config import Config
from .pipeline import UniAVPipeline
from .report import compare_table, format_events, load_gt, show

__all__ = ['Config', 'UniAVPipeline', 'load', 'describe_video', 'describe_features', 'search', 'embed_text',
           'show', 'compare_table', 'format_events', 'load_gt']

_pipeline = None


def load(**config):
    """(Re)load the pipeline. Keyword arguments override Config fields, e.g. device='cpu'."""
    global _pipeline
    _pipeline = UniAVPipeline(Config.from_env(**config))
    return _pipeline


def _get():
    return _pipeline if _pipeline is not None else load()


def describe_video(path, video_id=None, store=True):
    """Video file -> {'video_id', 'duration', 'events': [{start, end, score, caption, similarity,
    consensus, alternatives, embedding (np.ndarray, 512)}], 'timing', 'device'}.
    With store=True the events are kept (in memory and in Config.index_dir) for search()."""
    return _get().process(path, video_id=video_id, store=store)


def describe_features(visual, audio, duration, video_id=None, store=True):
    """Precomputed ONE-PEACE features (T, 1536) each, as in data/youcookii/av_features -> same
    result as describe_video, without running the encoders. store=True keeps it for search()."""
    return _get().process_stored(visual, audio, duration, video_id or 'video', store=store)


def search(query, top_k=10, video_ids=None):
    """Text query -> best-matching events among stored videos: [{video_id, start, end, caption, score}]."""
    return _get().search(query, top_k=top_k, video_ids=video_ids)


def embed_text(text):
    """Text -> 512-d normalised vector (numpy) comparable by dot product with event embeddings."""
    return _get().embed_query(text).cpu().numpy()
