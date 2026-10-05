---
license: other
pretty_name: COIN videos and InternVideo2 + BEATs features
tags:
- video
- audio
- instructional-videos
- temporal-localization
- coin
---

# COIN: videos and InternVideo2 + BEATs features (private, work in progress)

[COIN](https://coin-dataset.github.io/) (Tang et al., CVPR 2019) contains 11,827 YouTube
instructional videos covering 180 tasks in 12 domains, for 468 hours in total. It has
46,354 step segments with timestamps and a step label, for example `remove the old memory chip`.
This repo holds the videos we could still download in 2026 and one feature vector per second.
The features are extracted **exactly like our YouCook2 features**
(`nguyenminh04/uniav-youcook2-data`, `iv2_feats/`), so the two datasets can be trained together.

Code: [`tools/coin_hub.py`](https://github.com/im-xiaoming/UniAV-fixed/blob/coin-extract/tools/coin_hub.py)
and `tools/extract_internvideo2.py` on branch `coin-extract` of `im-xiaoming/UniAV-fixed`.
Status: the videos are downloaded and the features are extracted shard by shard by several
GPUs (a Colab T4 and 2 Kaggle T4). See `videos/manifest.json` and the `feats/` folder.

## Layout

| Path | Content |
|---|---|
| `feats/coin_feats_NNNN.tar` | **features**: one `<youtube_id>.npz` per video of shard NNNN, plus `failed.txt` (videos the extractor could not read; usually empty) |
| `videos/coin_videos_NNNN.tar` | 100 source videos per shard, `<youtube_id>.mp4`, at most 480p, video + audio (yt-dlp) |
| `videos/manifest.json` | `{"NNNN": [youtube_id, ...]}`: which videos are in which shard |
| `videos/DONE` | written once the download is complete |
| `claims/NNNN__<worker>` | bookkeeping: which GPU worker took which shard |
| `text_feats/coin_label_emb_iv2.npz` | InternVideo2 text vectors (512-d) of the 749 step labels: `labels`, `emb` (centred on the COIN label mean, pairwise cosine 0.01), `raw` (normalised, uncentred), `mean`, `mean_youcook2`. Same text tower as YouCook2's `caption_emb_iv2.npz`; made by `tools/embed_captions.py --coin` |
| `parity/6uHoTJSLoL8.mp4` | a YouCook2 video used to check that every worker reproduces the YouCook2 features (cosine 1.00000) |

About 18% of COIN videos are no longer available on YouTube (deleted or private), so roughly 9,700 videos are expected.
The annotations are not copied here. They come from [coin-dataset/annotations](https://github.com/coin-dataset/annotations) (`COIN.json`), keyed by the same YouTube id.

## Feature format

Each `.npz` holds three float16 arrays with one row per second of video, with **row i centred on i + 0.5 s**:

| Key | Shape | What |
|---|---|---|
| `v768` | (T, 768) | InternVideo2-Stage2-1B attention-pooled vision embedding of 4 frames (2 fps, 224×224) around second i; raw norm ~50 |
| `v512` | (T, 512) | `vision_proj(v768)`, L2-normalised, in InternVideo2's video-text space (its BERT-large text tower embeds sentences in the same space) |
| `a768` | (T, 768) | BEATs (from InternVideo2-Stage2-6B-Audio, `audio_6b.pth`) on a 3 s window of 16 kHz mono audio centred on second i, mean-pooled; zeros if the video has no audio |

Checkpoints: `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` (gated) and `OpenGVLab/InternVideo2-Stage2-6B-Audio`, fp16.
Our models L2-normalise every row before use (`iv2_l2norm: true`).

## Samples

The first videos of `feats/coin_feats_0000.tar`, with their COIN annotation:

| youtube id | task (COIN class) | duration | rows | steps (seconds: label) |
|---|---|---|---|---|
| `0R9pdc9dO3Q` | ReplaceBatteryOnTVControl | 122.4 s | 123 | 14–16 open cover · 16.5–19 remove battery · 108–113.5 put battery in · 114–121.5 close cover |
| `0xybXrdeSTk` | CutMango | 92.7 s | 93 | 5–12 cut the wide side · 21–34 remove the core · 54–58 cut the wide side · 60–72 remove the core |
| `1nQp_Yb8YW0` | ReplaceMemoryChip | 165.6 s | 166 | 64–73 take out the shell · 74–90 remove the old memory chip · 95–125 install the new memory chip · 126–135 fit on the shell |
| `2zqpJb-3SyI` | CutGrapeFruit | 174.9 s | 175 | 126–128 cut in half · 131–133 slice the pulp |

```python
import json, tarfile, numpy as np
from huggingface_hub import hf_hub_download

tar = hf_hub_download('nguyenminh04/coin-data', 'feats/coin_feats_0000.tar', repo_type='dataset')
tarfile.open(tar).extractall('coin_feats')
z = np.load('coin_feats/0R9pdc9dO3Q.npz')
{k: z[k].shape for k in z.files}   # {'v768': (123, 768), 'v512': (123, 512), 'a768': (123, 768)}

coin = json.load(open('COIN.json'))['database']                    # from coin-dataset/annotations
for a in coin['0R9pdc9dO3Q']['annotation']:
    start, end = a['segment']                                      # seconds
    rows = z['v768'][int(start):int(end) + 1]                      # features of that step
    print(a['label'], rows.shape)
```

## Differences from the YouCook2 features

- **Same script, same settings, same checkpoints:** every worker reproduces the stored YouCook2 features of `6uHoTJSLoL8` with cosine 1.00000 for all 182 rows (on Colab torch 2.11 and Kaggle torch 2.0).
- **The source encode differs:** YouTube serves a different encode than the one YouCook2 was downloaded from. Re-downloading a YouCook2 video with yt-dlp's default format gives video cosine ~0.997 (min 0.989) to the stored features, and audio is identical.
- **COIN labels are short step names, not sentences:** 749 distinct labels of 4.9 words on average, each reused across videos of the same task. YouCook2 captions average 8.8 words and are nearly all unique.

## License

COIN annotations and videos belong to their authors and to YouTube uploaders: research use only, under COIN's terms. This repo is private and is not for redistribution.
