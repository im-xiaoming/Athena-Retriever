"""UniAV event captioning as Python functions.

    import athena as uv

    uv.query(video='cooking.mp4')                       # video: events with captions
    uv.query(video='cooking.mp4', text='cut the onion') # video + sentence: where it happens in it
    uv.query(text='cut the onion')                      # sentence: where it happens in stored videos

    result = uv.describe_video('cooking.mp4')          # events with captions and embeddings
    result = uv.describe_sample('6uHoTJSLoL8')         # or a shipped sample: features already extracted
    for e in result['events']:
        print(e['start'], e['end'], e['caption'])

    uv.show(result)                                    # predicted events next to the YouCook2 GT
    uv.plot(result)                                    # the same as a timeline figure
    hits = uv.search('cut the onion', top_k=5)         # over every video described so far
    m = uv.ground('6uHoTJSLoL8', 'add soy sauce')      # one video (stored id, sample id or file)
    vec = uv.embed_text('add salt to the pot')         # 512-d query vector, same space as events

The model and encoders load on first use (GPU if available: CUDA or Apple MPS; else CPU).
Use uv.load(...) to pick a device or checkpoint explicitly, or AthenaPipeline for full control.
"""
from .config import Config
from .pipeline import AthenaPipeline
from .report import compare_table, format_events, load_gt, plot, show

__all__ = ['Config', 'AthenaPipeline', 'load', 'query', 'ground', 'describe_video', 'describe_features', 'describe_sample', 'samples',
           'search', 'embed_text', 'show', 'plot', 'compare_table', 'format_events', 'load_gt']

_pipeline = None


def load(**config):
    """(Re)load the pipeline. Keyword arguments override Config fields, e.g. device='cpu'."""
    global _pipeline
    _pipeline = AthenaPipeline(Config.from_env(**config))
    return _pipeline


def _get():
    return _pipeline if _pipeline is not None else load()


def describe_video(path, video_id=None, store=True):
    """Video file -> {'video_id', 'duration', 'events': [{start, end, score, caption, similarity,
    consensus, alternatives, embedding (np.ndarray, 512)}], 'timing', 'device'}.
    With store=True the events are kept (in memory and in Config.index_dir) for search()."""
    return _get().process(path, video_id=video_id, store=store)


def describe_features(visual, audio, duration, video_id=None, store=True):
    """Precomputed features (T, C) in the format of the loaded checkpoint -> same result as
    describe_video, without running the encoders. For the InternVideo2 model: visual = v768
    (one row per second, as in data/youcookii/iv2_feats/<id>.npz), audio = a768. store=True keeps
    the result for search()."""
    return _get().process_stored(visual, audio, duration, video_id or 'video', store=store)


def samples():
    """Ids of the sample videos shipped with features in athena/samples (YouCook2 validation)."""
    return _get().samples()


def describe_sample(video_id, store=True):
    """A shipped sample (see samples()) -> same result as describe_video, in about a second."""
    return _get().process_sample(video_id, store=store)


def query(video=None, text=None, top_k=5):
    """The production entry point: a video, a video and a sentence, or a sentence alone.
    See AthenaPipeline.query."""
    return _get().query(video=video, text=text, top_k=top_k)


def ground(video, text, top_k=3):
    """Video (stored id, sample id or file) + sentence(s) -> {'video_id', 'query', 'matches':
    [{start, end, score, caption}]}: where in that video the sentence happens."""
    return _get().ground(video, text, top_k=top_k)


def search(query, top_k=10, video_ids=None):
    """Sentence -> best-matching segments among stored videos: [{video_id, start, end, caption, score}]."""
    return _get().search(query, top_k=top_k, video_ids=video_ids)


def embed_text(text):
    """Text -> 512-d normalised vector (numpy) comparable by dot product with event embeddings."""
    return _get().embed_query(text).cpu().numpy()
