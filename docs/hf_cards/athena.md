---
license: other
pretty_name: Athena checkpoints
library_name: pytorch
tags:
- video
- audio
- dense-video-captioning
- temporal-localization
- video-retrieval
- youcook2
- coin
---

# Athena: checkpoints

Weights of Athena, a model that finds the steps of an instructional video, captions each one and
grounds a sentence in time. Input is a video's InternVideo2 + BEATs features (one row per second),
optionally with a sentence. Code, architecture and training:
[im-xiaoming/Athena-Retriever](https://github.com/im-xiaoming/Athena-Retriever)
(`docs/ARCHITECTURE.md`, `athena/README.md`).

| File | Run | Trained on | Size |
|---|---|---|---|
| `athena.pth` | `ov_E1` | YouCook2 | 278 MB |
| `athena_coin.pth` | `ov_R1` | YouCook2 + COIN | 278 MB |
| `internvideo2/InternVideo2-stage2_1b-224p-f4.pt` | | OpenGVLab's video encoder and text tower, unchanged | 2.8 GB |
| `internvideo2/audio_6b.pth` | | OpenGVLab's audio encoder (BEATs), unchanged | 361 MB |

Both are API checkpoints made by `tools/export_api_ckpt.py`: fp16 weights, the training config,
the caption pool, and a second weight set (`state_dict_seg`, the best segmentation epoch) used for
segmentation and grounding. Captions are chosen from the 8218 YouCook2 training sentences in
InternVideo2 text space (`caption_space: iv2`).

## Results (YouCook2 validation, 394 videos)

Numbers stored in each checkpoint (`final_eval`), main weights.

| | `athena.pth` | `athena_coin.pth` |
|---|---|---|
| R@0.3 / R@0.5 / R@0.7 | 72.6 / 53.6 / 30.9 | 70.7 / 52.2 / 31.3 |
| mIoU | 49.9 | 49.4 |
| CIDEr (consensus pick) | 88.2 | 87.0 |
| Ground R1@0.5 / R1@0.7 | 47.3 / 29.2 | 45.8 / 28.9 |
| COIN seen-task Acc | 5.4 | 48.9 |
| COIN unseen-task Acc | 7.6 | 9.1 |

`athena.pth` is the default. `athena_coin.pth` has seen COIN tasks, so it classifies them far better
but is slightly behind on YouCook2 captions.

## Use

```bash
git clone https://github.com/im-xiaoming/Athena-Retriever && cd Athena-Retriever
huggingface-cli download nguyenminh04/athena athena.pth --local-dir ckpt/api
# only to encode new videos; the encoders go in ckpt/internvideo2/
huggingface-cli download nguyenminh04/athena --include "internvideo2/*" --local-dir ckpt
```

```python
import athena as uv
r = uv.describe_sample('6uHoTJSLoL8')     # a validation video shipped with its features
r = uv.describe_video('cooking.mp4')      # any video: needs the InternVideo2 + BEATs encoders
uv.query(video='cooking.mp4', text='cut the onion')
```

To use `athena_coin.pth`, set `ATHENA_CHECKPOINT=ckpt/api/athena_coin.pth`. Encoding a new video
needs the InternVideo2 encoders in `internvideo2/` (copies of `OpenGVLab/InternVideo2-Stage2_1B-224p-f4`
and `OpenGVLab/InternVideo2-Stage2-6B-Audio`, Apache-2.0); stored features need no encoder.

## Limits

Trained on cooking and COIN instructional videos; other domains are untested. Checkpoints load
with `torch.load(..., weights_only=False)` and need numpy 2.x, so load them only from this repo.
