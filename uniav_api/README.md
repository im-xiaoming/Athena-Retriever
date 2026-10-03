# uniav_api

Python functions that turn a cooking video into timed events, each with a caption and a
512-d vector for text search. It is a self-contained inference copy of the event model in
`libs/modeling/event_archs.py`, plus the ONE-PEACE encoders, rewritten so it needs no fairseq,
no compiled extension and no separately installed ffmpeg. It runs on Windows, Linux and macOS.

```python
import uniav_api as uv

result = uv.describe_video('cooking.mp4')
for e in result['events']:
    print(e['start'], e['end'], e['caption'])      # seconds, seconds, best caption

uv.search('boil the noodles', top_k=5)              # events of every video described so far
uv.embed_text('add salt')                           # 512-d query vector, same space as e['embedding']

# precomputed ONE-PEACE features (data/youcookii/av_features), no encoders needed
uv.describe_features(visual, audio, duration, video_id='abc')
```

Each event holds `start`, `end`, `score`, `caption`, `similarity` (cosine to the caption),
`consensus`, `alternatives` (3 other candidate captions) and `embedding` (numpy, 512).
Results are kept in `uniav_api/index/<video_id>.json` so `search()` covers earlier videos.

## How it works

1. **Encode** (`encoders/`): the same features the model was trained on.
   - video: 16 fps, 256x256 centre crop, 16-frame windows every 0.5 s through the
     Kinetics-400 ONE-PEACE backbone -> (T, 1536)
   - audio: 16 kHz, 1 s windows every 0.5 s through ONE-PEACE's audio branch -> (T, 1536)
2. **Segment** (`model/event_model.py`): features resized to 256 steps, the event / boundary /
   IoU heads, then Gaussian soft-NMS (`postprocess.py`, a numpy port of the C++ op).
3. **Select**: events with score >= `min_score` (0.40), at most 0.3 IoU with a better event.
4. **Caption** (`captioner.py`): each event vector against the 8218 train captions, consensus pick.
5. **Search**: a query goes through ONE-PEACE text and the model's caption projection, then cosine.

## Checks done (2026-10-04)

| Check | Result |
|---|---|
| API audio features vs training features (same video) | cosine 1.00000 |
| API video features vs training features | cosine mean 0.99996, min 0.9989 (fp16) |
| numpy soft-NMS vs the C++ extension, 200 random cases | identical |
| Model parity, val set, k = #GT | R@0.5 48.70 (training eval 48.70) |
| Event selection at min_score 0.40, val set | F1@IoU0.5 0.476, 8.5 events/video (GT 7.7) |

`python -m uniav_api.calibrate` repeats the last two; `python -m uniav_api.demo` prints
events next to the annotations for 10 validation videos and runs a few searches.

## Devices and memory

`Config.device='auto'` picks CUDA, then Apple MPS, then CPU. Encoders run in fp16 on GPUs
and fp32 on CPU; the event model always runs fp32. Override any setting with
`uv.load(device='cpu', checkpoint=...)` or an environment variable `UNIAV_<FIELD>`.

| | Video encoder | Audio encoder | Text encoder (search) |
|---|---|---|---|
| parameters | 1.66B | 1.53B | 1.59B |
| fp16 memory | ~3.3 GB | ~3.1 GB | ~3.2 GB |

With less than 10 GB of GPU memory, or on MPS / CPU, the video and audio encoders are
loaded one at a time and released after use. Speed is dominated by the video encoder:
on an RTX 3060 an 8-minute video takes about 15 minutes to encode; the model itself takes
under a second. CPU works but is much slower.

## Files needed

| Path | What |
|---|---|
| `ckpt/omni65/best_cap.pth.tar` | event model (best captioning run, see experiments/RESULTS.md) |
| `configs/youcook2_event.yaml` | model and feature-grid settings |
| `uniav_api/assets/caption_pool.npz` | train captions + ONE-PEACE vectors (built from caption_emb.npz) |
| `ONEPEACE_extract_embd_code/models/onepeace_video_k400.pth` | video encoder |
| `ONEPEACE_extract_embd_code/models/one-peace-audio.pt` | audio encoder |
| `ONEPEACE_extract_embd_code/models/one-peace-text.pt` | text encoder (only for search / embed_text) |

The encoder folder can be swapped: `encoders/base.py` defines the interface, so an
InternVideo2 encoder can replace ONE-PEACE once a model is trained on those features.
