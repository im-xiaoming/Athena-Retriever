---
license: other
pretty_name: UniAV on YouCook2 - features, teacher vectors and checkpoints
tags:
- video
- audio
- dense-video-captioning
- temporal-localization
- youcook2
---

# UniAV on YouCook2: features, teacher vectors and checkpoints (private)

This repo holds everything needed to train and run the YouCook2 event segmentation + captioning
model in [`im-xiaoming/UniAV-fixed`](https://github.com/im-xiaoming/UniAV-fixed). The model finds
the steps of an uncut cooking video and writes one caption per step.

YouCook2 (Zhou et al., AAAI 2018) has 2,000 cooking videos of 89 recipes, with step segments and
one English sentence per step. 1,500 of the train + validation videos are still downloadable:
1,106 are used for training and 394 for validation. Annotations and caption vectors are in the
git repo (`data/youcookii/`), not here.

## Layout

| Path | Size | Content |
|---|---|---|
| `iv2_feats/iv2_feats_00..07.tar` + `manifest.json` | 1.9 GB | **main input features**: one `<youtube_id>.npz` per video (1,500), 1 row per second |
| `iv2_feats_shift/iv2_feats_shift_00..07.tar` + `manifest.json` | 1.9 GB | the same features with every window moved by +0.5 s. Interleaving the two gives 2 rows per second (`tools/make_iv2_dense.py`) |
| `teacher/omni_emb_full.npz` | 132 MB | OmniRetriever-7B embeddings (3584-d, float16): `<video>#<step>__av` for the audio+video clip of a step (9,060), `<video>#<step>__text` for its caption (8,718) |
| `text_feats/caption_emb_iv2.npz` | 11 MB | **InternVideo2 text vectors** (512-d) of every caption, in the space of `v512`; centred on the train-caption mean (`mean` stored) because raw vectors are anisotropic (pairwise cosine 0.95). Keys `"<video>#<step>"`, plus `sentences`. The caption space of the model and of the `txt_sim` metric |
| `api/uniav_iv2.pth` | 263 MB | **legacy** API checkpoint of run `iv2`, trained in the removed ONE-PEACE caption space; only the `test`-branch Colab demo still loads it. Its replacement, trained in InternVideo2 caption space, will be `api/uniav_civ2_v512.pth` |
| `api/capgen_prefix.pth` | 251 MB | GPT-2 caption generator (prefix variant) trained on `uniav_iv2.pth` segments (legacy, same as above) |

## Feature format (`iv2_feats`)

Each `.npz` holds three float16 arrays, with **row i centred on i + 0.5 s**:

| Key | Shape | What |
|---|---|---|
| `v768` | (T, 768) | InternVideo2-Stage2-1B attention-pooled vision embedding of 4 frames (2 fps, 224×224) around second i |
| `v512` | (T, 512) | `vision_proj(v768)`, L2-normalised, in InternVideo2's video-text space |
| `a768` | (T, 768) | BEATs (InternVideo2-Stage2-6B-Audio) on a 3 s window of 16 kHz mono audio centred on second i |

Extracted with `tools/extract_internvideo2.py`; text vectors with `tools/embed_captions.py` (InternVideo2's BERT-large text tower, first 19 layers + `text_proj`). The COIN features in `nguyenminh04/coin-data` use the same script, settings and checkpoints.

## Samples

`6uHoTJSLoL8` (validation, 181.9 s, 6 steps) → `v768` (182, 768), `v512` (182, 512), `a768` (182, 768):

| step | seconds | caption |
|---|---|---|
| 0 | 53–73 | heat chili oil in pan with ginger garlic chili paste and schizuan paste |
| 1 | 74–92 | add mince pork and stir |
| 2 | 93–103 | add rice wine to pan |
| 3 | 104–122 | add soy sauce to pan |

```python
import tarfile, numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

d = snapshot_download('nguyenminh04/uniav-youcook2-data', repo_type='dataset',
                      allow_patterns=['iv2_feats/*', 'teacher/*'])
for i in range(8):
    tarfile.open(f'{d}/iv2_feats/iv2_feats_0{i}.tar').extractall('data/youcookii/iv2_feats')
z = np.load('data/youcookii/iv2_feats/6uHoTJSLoL8.npz')
step0 = z['v768'][53:74]                                   # rows of step 0 (53-73 s)

t = np.load(f'{d}/teacher/omni_emb_full.npz')
t['6uHoTJSLoL8#0__av'].shape, t['6uHoTJSLoL8#0__text'].shape   # (3584,), (3584,)
```

ONE-PEACE (features, text encoder, caption vectors) was removed on 2026-10-05.

## Legacy checkpoint `api/uniav_iv2.pth`

`{'state_dict', 'config', 'run': 'iv2', 'commit', 'epoch': 7, 'final_eval'}`. It is loaded by
`uniav_api` (`UniAVPipeline`), which also reads checkpoints saved before the 2026-10-05 cleanup.
YouCook2 validation, 394 videos:

| R@0.5 | R@0.7 | ret_sim (ONE-PEACE) | CIDEr (retrieved) | METEOR | CIDEr (generated, `capgen_prefix`) |
|---|---|---|---|---|---|
| 54.3 | 31.2 | 0.754 | 87.9 | 15.56 | 95.1 |

## License

YouCook2 videos and annotations belong to their authors and to YouTube uploaders: research use only. This repo is private and is not for redistribution.
