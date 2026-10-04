# uniav_api

Python functions that turn a cooking video into timed steps, each with a caption chosen from
the 8218 YouCook2 training sentences and a 512-d vector for text search. It is a self-contained
inference copy of the event model in `libs/modeling/event_archs.py` (architecture:
`docs/UniAV_new.drawio.svg`, walkthrough: `docs/UniAV_new_walkthrough.md`).

Colab demo: `uniav_api/demo_colab.ipynb` (samples need no GPU and no encoder).

```python
import uniav_api as uv

uv.samples()                                   # YouCook2 val videos shipped with their features
r = uv.describe_sample('6uHoTJSLoL8')          # ~1 s, no encoder needed
uv.show(r)                                     # each event next to the real step it matches
uv.plot(r)                                     # the same as a timeline figure (matplotlib)
uv.compare_table(r)                            # the same as a pandas DataFrame

r = uv.describe_video('cooking.mp4')           # any video: runs InternVideo2 + BEATs first
for e in r['events']:
    print(e['start'], e['end'], e['caption'])

uv.search('boil the noodles', top_k=5)         # events of every video described so far
uv.describe_features(v768, a768, duration)     # your own stored features (one row per second)
```

Each event holds `start`, `end` (seconds), `score`, `caption`, `similarity`, `consensus`,
`alternatives` (3 other candidate captions) and `embedding` (numpy, 512). Results are kept in
`uniav_api/index/<model>/<video_id>.json` so `search()` covers earlier videos.

## Model

`ckpt/api/uniav_iv2.pth` (263 MB, fp16), exported from run `iv2` by `tools/export_api_ckpt.py`. It
carries its training config, so the API knows the features it takes. YouCook2 validation, 394 videos:

| | R@0.5 | R@0.7 | ret_sim | CIDEr | METEOR |
|---|---|---|---|---|---|
| this model (InternVideo2 + BEATs) | 54.3 | 31.2 | 0.754 | 87.9 | 15.6 |
| same setup, second seed | 52.8 | 29.6 | 0.758 | 89.9 | 15.5 |
| previous API model `omni65` (ONE-PEACE) | 48.7 | 25.3 | 0.758 | 86.5 | 15.4 |

R@0.5 counts the real steps covered by a predicted event with IoU >= 0.5 when the model keeps as
many events as there are real steps. In the API, events are kept by score instead (`min_score`
0.36, at most 0.3 IoU between kept events): F1 at IoU 0.5 is 0.534, 8.4 events per video (real 7.7).

## Samples

`uniav_api/samples/<id>.npz`: InternVideo2 v768 / v512, BEATs a768 (one row per second, fp16) and
the duration. All are validation videos, never seen in training.

| id | dish | length | F1 at IoU 0.5 |
|---|---|---|---|
| `-Ju39A-G0Dk`, `6uHoTJSLoL8`, `W2gnFLOi_AQ` | chili, mapo tofu, salmon | 8:16, 3:02, 6:08 | typical (the raw mp4s are in `data/demo_videos`) |
| `XEifm-iXMvs` | falafel with tzatziki | 4:51 | 0.94 |
| `9GX8f5EwwE4` | oven-baked dish, 13 steps | 4:02 | 0.85 |
| `e8S1vFC8zYk` | mac and cheese | 2:33 | 0.89 |
| `SOMsxGGSTUk` | Thai noodles | 5:20 | 0.89 |
| `cMzyB4m3VHY` | pizza | 3:41 | 0.83 |

The last five were picked among the better videos (median F1 over the validation videos of
2.5 to 5.5 minutes is 0.57); the first three were chosen before the model existed.

## How it works

1. **Encode** (`encoders/internvideo2.py`, reusing `tools/extract_internvideo2.py`): 2 fps,
   224x224, one 4-frame window per second through InternVideo2-1B (v768); 16 kHz audio, a 3 s
   window per second through BEATs (a768). Rows are L2-normalised.
2. **Segment** (`model/event_model.py`): features resized to 256 steps, the event / boundary /
   IoU heads, then Gaussian soft-NMS (`postprocess.py`, a numpy port of the C++ op).
3. **Select**: events with score >= `min_score`, at most `max_overlap` IoU with a better event.
4. **Caption** (`captioner.py`): each event vector against the train captions, consensus pick
   (MBR over the top 20). Only the caption space is used; adding the OmniRetriever teacher
   space changed ret_sim by at most 0.003 on validation.
5. **Search**: with the ONE-PEACE text encoder (6 GB, optional) the query goes through it and
   the model's caption projection. Without it, the query is matched to the closest train
   captions by word overlap (TF-IDF) and their vectors are averaged: phrase queries like
   recipe steps.

## Checks done (2026-10-04)

| Check | Result |
|---|---|
| Model parity, val set, k = #GT (`python -m uniav_api.calibrate`) | R@0.5 54.27 (training eval 54.27), fp16 checkpoint |
| Event selection at min_score 0.36, val set | F1@IoU0.5 0.534, 8.4 events per video |
| numpy soft-NMS vs the C++ extension, 200 random cases | identical |
| `describe_video` encoder vs the Colab features, `6uHoTJSLoL8` (182 s) | cosine 1.00000 (mean and min) for v768, v512, a768; 42 s on an RTX 3060 including model loading, peak GPU 4.3 GB |

`python -m uniav_api.demo` prints every sample next to its annotations and runs a few searches.

## Files needed

| Path | What | For |
|---|---|---|
| `ckpt/api/uniav_iv2.pth` | event model; HF dataset `nguyenminh04/uniav-youcook2-data`, file `api/uniav_iv2.pth` (private) | everything |
| `uniav_api/assets/caption_pool.npz` | train captions + ONE-PEACE text vectors (in git) | everything |
| `data/youcookii/annotations/youcookii_annotations_trainval.json` | GT steps (in git) | show / plot |
| `InternVideo/` | `git clone --depth 1 https://github.com/OpenGVLab/InternVideo` (unmodified) | describe_video |
| `ckpt/internvideo2/InternVideo2-stage2_1b-224p-f4.pt` | HF `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` (gated) | describe_video |
| `ckpt/internvideo2/audio_6b.pth` | HF `OpenGVLab/InternVideo2-Stage2-6B-Audio` | describe_video |
| `ONEPEACE_extract_embd_code/models/one-peace-text.pt` | ONE-PEACE text encoder, 6 GB | search, optional |

`describe_video` also needs `timm`, `einops`, `torchaudio` (`pip install -r uniav_api/requirements.txt`).
The older ONE-PEACE model still works: `uv.load(checkpoint='ckpt/omni65/best_cap.pth.tar')` with
`configs/youcook2_event.yaml` and the ONE-PEACE encoders in `ONEPEACE_extract_embd_code/models/`.

## Devices and memory

`Config.device='auto'` picks CUDA, then Apple MPS, then CPU; override with `uv.load(device='cpu')`
or an environment variable `UNIAV_<FIELD>`. The event model (131M parameters) runs fp32 anywhere
in under a second per video. The InternVideo2 encoders run fp16 on a GPU (about 3 GB of weights);
extraction ran at about 30 seconds of video per second on an A100.
