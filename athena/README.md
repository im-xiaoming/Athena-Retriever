# athena

Python functions that turn a cooking video into timed steps, each with a caption chosen from
the 8218 YouCook2 training sentences and a 512-d vector for text search. It is a self-contained
inference copy of the event model in `libs/modeling/event_archs.py` (architecture: see the
"Architecture" section of the top-level `README.md` and `docs/ARCHITECTURE.md`).

Colab demo: `athena/demo_colab.ipynb` (samples need no GPU and no encoder).

```python
import athena as uv

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

```python
uv.load(caption_mode='generate')               # captions written by GPT-2 instead of picked (see below)
```

## Three kinds of input (`uv.query`)

| Input | Call | Returns |
|---|---|---|
| a video | `uv.query(video='cooking.mp4')` (a file, a sample id or a stored id) | every step of the video with a caption, as `describe_video` |
| a video and a sentence | `uv.query(video='cooking.mp4', text='cut the onion')` = `uv.ground(...)` | where in that video the sentence happens: `matches` [{start, end, score, caption}], best first |
| a sentence | `uv.query(text='cut the onion')` = `uv.search(...)` | where it happens across every stored video: `hits` [{video_id, start, end, score, caption}] |

A sentence goes through InternVideo2's text tower and the model's **ground head**
(`loss_weight_ground` in training): the model reads the whole video again with the sentence as a
condition, so the answer does not have to be one of the steps found without it. `score` is the
model's probability that the segment is the event the sentence describes; `caption` is what the
model itself would call that segment. A sentence alone first ranks the stored videos by their event
vectors, then is grounded in the best 20 (`rerank`); every processed video keeps its features
(`index/<model>/<id>.npz`, fp16) for that. A checkpoint without a ground head (trained before
2026-10-05) answers both by ranking the stored events by cosine (`method: event cosine`).
A video file given with a sentence right after it was described is not encoded again.

Each event holds `start`, `end` (seconds), `score`, `caption`, `similarity`, `consensus`,
`alternatives` (3 other candidate captions) and `embedding` (numpy, 512). Results are kept in
`athena/index/<model>/<video_id>.json` so `search()` covers earlier videos.

## Model

`ckpt/api/athena.pth` is training run `ov_E1`, exported by `tools/export_api_ckpt.py`: InternVideo2
text space (`caption_space: iv2`) with the ground head. It carries two weight sets: the main weights
(epoch 10, best captions) and `state_dict_seg` (epoch 7, best segmentation); `use_seg_weights` picks
segmentation and grounding from the second and captions from the first. YouCook2 validation,
394 videos, main weights, as stored in the checkpoint (`final_eval`):

| Measure | Value |
|---|---|
| R@0.3 / R@0.5 / R@0.7 | 72.6 / 53.6 / 30.9 |
| mIoU | 49.9 |
| CIDEr (consensus pick, `mbr_student`) | 88.2 |
| Ground R1@0.5 / R1@0.7 | 47.3 / 29.2 |

R@k counts the real steps covered by a predicted event with IoU >= k when the model keeps as many
events as there are real steps (7.7 per video). In the API, events are kept by score instead
(`min_score` 0.40, at most 0.3 IoU between kept events). `ckpt/api/athena_coin.pth` (run `ov_R1`,
trained with COIN) is not used by the API yet.

## Samples

`athena/samples/<id>.npz`: InternVideo2 v768 / v512, BEATs a768 (one row per second, fp16) and
the duration. All are validation videos, never seen in training.

| id | dish | length | F1 at IoU 0.5 |
|---|---|---|---|
| `-Ju39A-G0Dk`, `6uHoTJSLoL8`, `W2gnFLOi_AQ` | chili, mapo tofu, salmon | 8:16, 3:02, 6:08 | typical (the raw mp4s are in `data/demo_videos`) |
| `XEifm-iXMvs` | falafel with tzatziki | 4:51 | 0.94 |
| `9GX8f5EwwE4` | oven-baked dish, 13 steps | 4:02 | 0.85 |
| `e8S1vFC8zYk` | mac and cheese | 2:33 | 0.89 |
| `SOMsxGGSTUk` | Thai noodles | 5:20 | 0.89 |
| `cMzyB4m3VHY` | pizza | 3:41 | 0.83 |

F1 values are from the earlier checkpoint. The last five were picked among the better videos (median F1 over the validation videos of
2.5 to 5.5 minutes is 0.57); the first three were chosen before the model existed.

## How it works

1. **Encode** (`encoders/internvideo2.py`, reusing `tools/extract_internvideo2.py`): 2 fps,
   224x224, one 4-frame window per second through InternVideo2-1B (v768); 16 kHz audio, a 3 s
   window per second through BEATs (a768). Rows are L2-normalised.
2. **Segment** (`model/event_model.py`): features resized to 256 steps, the event / boundary /
   IoU heads, then Gaussian soft-NMS (`postprocess.py`, a numpy port of the C++ op).
3. **Select**: events with score >= `min_score`, at most `max_overlap` IoU with a better event.
4. **Caption** (`captioner.py`): each event vector against the train captions, consensus pick
   (MBR over the top 20), in InternVideo2 text space. Adding the OmniRetriever teacher space
   changed caption similarity by at most 0.003 on validation, so only the caption space is used.
5. **Search**: the query goes through InternVideo2's text tower (read from the InternVideo2 video
   checkpoint, centred like the caption vectors) and the model's caption projection. Without that
   checkpoint, the query is matched to the closest train captions by word overlap (TF-IDF) and
   their vectors are averaged: phrase queries like recipe steps.

## Generated captions (`caption_mode='generate'`)

`athena/generator.py`: GPT-2 small fine-tuned to write a caption from the event's vector q and
8 level-0 feature tokens sampled across the event (a ClipCap-style prefix), trained on the YouCook2
train steps of this model (`tools/capgen/`). The retrieved train caption stays in
`retrieved_caption`. Checkpoint `ckpt/api/capgen_prefix.pth` (251 MB, fp16); GPT-2's tokenizer and
config download from HuggingFace (`openai-community/gpt2`) on first use; needs `transformers`.

| YouCook2 val | caption | CIDEr | METEOR | BLEU-4 |
|---|---|---|---|---|
| GT steps (3031) | retrieved (default) | 88.9 | 15.13 | 7.02 |
| GT steps (3031) | generated | **97.6** | **15.72** | **8.41** |
| events found by the model, matched to a GT step (2253) | retrieved | 86.0 | 14.91 | 6.89 |
| events found by the model, matched to a GT step (2253) | generated | **95.1** | **15.42** | **7.93** |

Generated captions read naturally and score higher on average, but can invent details (e.g. "lamb"
for pork); retrieved captions are always real sentences. Check: on the GT steps of `6uHoTJSLoL8`
the API writes exactly the training script's captions (6/6), CPU and fp16 checkpoint.

## Checks done (2026-10-04, earlier checkpoint)

| Check | Result |
|---|---|
| Model parity, val set, k = #GT (`python -m athena.calibrate`) | R@0.5 54.27 (training eval 54.27), fp16 checkpoint |
| Event selection at min_score 0.36, val set | F1@IoU0.5 0.534, 8.4 events per video |
| numpy soft-NMS vs the C++ extension, 200 random cases | identical |
| `describe_video` encoder vs the Colab features, `6uHoTJSLoL8` (182 s) | cosine 1.00000 (mean and min) for v768, v512, a768; 42 s on an RTX 3060 including model loading, peak GPU 4.3 GB |

`python -m athena.demo` prints every sample next to its annotations and runs a few searches.

## Files needed

| Path | What | For |
|---|---|---|
| `ckpt/api/athena.pth` | event model; HF model `nguyenminh04/athena`, file `athena.pth` | everything |
| `athena/assets/caption_pool.npz` | train captions + InternVideo2 text vectors + centring mean (in git) | everything |
| `data/youcookii/annotations/youcookii_annotations_trainval.json` | GT steps (in git) | show / plot |
| `InternVideo/` | `git clone --depth 1 https://github.com/OpenGVLab/InternVideo` (unmodified) | describe_video |
| `ckpt/internvideo2/InternVideo2-stage2_1b-224p-f4.pt` | HF `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` (gated) | describe_video |
| `ckpt/internvideo2/audio_6b.pth` | HF `OpenGVLab/InternVideo2-Stage2-6B-Audio` | describe_video |
| `ckpt/api/capgen_prefix.pth` | caption generator, `tools/capgen/export_generator.py prefix` | caption_mode='generate' |

`describe_video` also needs `timm`, `einops`, `torchaudio` (`pip install -r athena/requirements.txt`).
Only models with InternVideo2 features and the InternVideo2 caption space (or the teacher's) are
supported; ONE-PEACE models need the code before 2026-10-05. Checkpoints saved before the cleanup
of that day (with unused AVEL / SED feed-forward layers, 135M parameters instead of 68M) load unchanged.

## Devices and memory

`Config.device='auto'` picks CUDA, then Apple MPS, then CPU; override with `uv.load(device='cpu')`
or an environment variable `ATHENA_<FIELD>`. The event model (131M parameters) runs fp32 anywhere
in under a second per video. The InternVideo2 encoders run fp16 on a GPU (about 3 GB of weights);
extraction ran at about 30 seconds of video per second on an A100.
