# Athena architecture

Source of truth: `libs/modeling/event_archs.py` (model, losses, inference), `libs/modeling/multimodal_backbones.py`
(backbone), `athena/` (production wrapper), `configs/youcook2_event.yaml` (sizes and weights quoted below).
Sizes are for the default config: 256 steps, `embd_dim = head_dim = 512`, 4 attention heads.

## 1. Inputs

| Input | Where it comes from |
|---|---|
| Visual `v768` | InternVideo2-Stage2-1B (frozen), 4 frames at 2 fps per 1 s window, one 768-d vector per second (`v512`, its projection into the text space, is optional: `iv2_video_keys`) |
| Audio `a768` | BEATs from InternVideo2-Stage2-6B-Audio (frozen), 3 s window per second, 768-d |
| Sentence (optional) | InternVideo2 text tower (frozen, BERT-large, 512-d); used by the ground head and for the caption pool |

Features come from `tools/extract_internvideo2.py` (see `docs/README.md` for the extraction pipeline). Each row is
L2-normalised and the video is resampled to **256 steps**, whatever its length, so one step is `duration / 256` seconds.

## 2. Overview

```
frames ─> InternVideo2 (frozen) ─> v768 (T x 768) ─┐
audio  ─> BEATs         (frozen) ─> a768 (T x 768) ─┤ L2 norm, resample to 256 steps
                                                    v
                    ConvTransformerBackbone (from UniAV)
                    V and A streams, 6 pyramid levels (strides 1, 2, 4, 8, 16, 32)
                    per level: Cat(V, A) -> 1024-d, 504 points in total
                                                    |
     ┌───────────────┬───────────────┬──────────────┴─┬─────────────────────────┐
     v               v               v                v                         v
 Event head     Boundary head    Embed head     (level-0 raw       Ground head  <── sentence (text tower)
 1 logit        2 offsets +      512-d / step    embed features)    1 logit / step, FiLM by the sentence
                1 IoU logit                          |
     └────── score, boundaries ──┐                   v
                                 v          SegmentContext(span, video mean, position)
              soft-NMS -> events (t_s, t_e) ────────> segment vector q (512-d)
                                                     |
                                    caption = pick from the caption pool by cosine + consensus,
                                    or text written by a GPT-2 generator (optional)
```

Three ways to use it (`athena.query`):

| Input | Path |
|---|---|
| video | events from event + boundary heads, each captioned from its vector `q` |
| video + sentence | ground head scores every step for that sentence, segment from the boundary head at the best step |
| sentence | rank stored videos by their event vectors, ground the sentence in the best 20 |

## 3. Backbone (`ConvTransformerBackbone`, unchanged from UniAV apart from the removed unused MLPs)

`backbone_arch = (2, 3, 5)`, `scale_factor = 2`, one copy per stream (V, A) unless noted.

1. **Embedding:** 2 masked Conv1D (k = 3) + LayerNorm + ReLU, 768 -> 512. Sinusoidal absolute positions added.
2. **Stem:** 2 self-attention blocks per stream (masked multi-head conv attention with depthwise convs, 4 heads, MLP x4, DropPath 0.1).
3. **Level 0, cross-attention:** V queries A and A queries V (`pyramid_attn: cross`; `self` makes each stream attend to itself).
4. **Pyramid:** 5 more cross-attention blocks per stream, each downsampling by 2 -> 6 levels with 256, 128, 64, 32, 16, 8 steps.
5. Output per level: V (512) and A (512), concatenated to **1024** channels for the heads.

Regression ranges per level (in steps): [0,4], [4,8], [8,16], [16,32], [32,64], [64,inf). A point is positive for a
ground-truth segment when its centre lies inside it and the segment length fits the level's range.

## 4. Heads (shared across the six levels; 3 layers = 2 Conv1D + LN + ReLU, then an output conv)

| Head | Output per point | Notes |
|---|---|---|
| Event | 1 logit: is there an event here | class agnostic; bias initialised for prior 0.01 |
| Boundary | 2 offsets (distance to start / end, in stride units, ReLU, learnable per-level `Scale`) + 1 IoU-quality logit | offset bias starts at 1.0 so no side is dead at init |
| Embed | 512-d unit vector | extra `vis_proj` conv; its level-0 raw features (512 x 256) feed `SegmentContext`; owns the learnable `logit_scale` |
| Ground | 1 logit for (point, sentence) | sentence 512-d -> FiLM (scale, shift, zero init) after the first conv layer; adds `cos_scale * cos(step embedding, sentence)` to the logit, so it starts from what the embed head knows. First layer runs once per video |

**SegmentContext:** for a span [s, e] on level 0: `span` = mean of the raw embed features over the span (cumulative sums),
`glob` = mean over the video, `pos` = (s/T, e/T). `q = L2( span + MLP([span, glob, pos]) )` with MLP 1026 -> 512 -> 512
whose last layer starts at zero, so `q` begins as the span mean.

## 5. Text side and caption pool

- **Caption pool:** the unique train captions (YouCook2 8,218; COIN step labels when COIN is used) as InternVideo2 text
  vectors, centred on a joint mean (`caption_emb_iv2j.npz`; raw vectors have mean pairwise cosine 0.95). Each dataset has its own pool
  rows (`pool_src`), plus an "other" text for background steps.
- `text_proj: linear` (config default): `clip_proj` (Linear 512 -> 512, lr 1e-3) maps pool vectors into the event space.
  `text_proj: none`: the text vectors stay frozen and only the video side learns to land in them (OV-AVEL style).
- **Teacher (training only, optional):** OmniRetriever-7B gives a 3584-d audio+video vector per GT clip and a text vector per
  caption. `omni_proj` (512 -> 1024 -> 3584) maps `q` there. At inference its scores can be added.

## 6. Training losses (weights from the config)

| Loss | What | Weight |
|---|---|---|
| `ev_loss` | sigmoid focal, event vs not, normalised by a running count of positives | 1 |
| `reg_loss` | DIoU on the two boundary distances at positive points | 1 |
| `iou_loss` | BCE of the IoU-quality logit against the true IoU of the (detached) prediction | 1 |
| `emb_loss` | per-step embedding at positive points vs the whole caption pool (`pool_loss`) | 0.2 |
| `span_loss` | `q` of every GT span and of a copy jittered by up to 20% of its length vs the pool | 0.2 |
| `omni_txt_loss` | `q` in the teacher space vs the teacher text pool | 0.2 |
| `omni_av_loss` | `q` in the teacher space must pick its own GT clip among all clips | 0.2 |
| `ground_loss` | focal; every caption of a video is a query for its own span; up to 2 captions of other videos (cosine < 0.7 to this video's captions) are queries with no span | 1 |
| `other_loss` | background steps pulled to the "other" text | 0 (off) |
| `modal_loss` | single-stream (V only, A only) vectors pulled to the fused vector and the caption | 0 (off) |

**`pool_loss`:** cross-entropy of a vector against the *whole* frozen caption pool (video-to-text InfoNCE-like, no queue, no memory bank,
no momentum encoder), logit scale learnable (init 1/0.07, clamp 100). Soft targets: `0.5` on the true caption and `0.5` spread by
caption-caption similarity (softmax, tau 0.02) so paraphrases are not punished like wrong captions. Rows only see captions of their own dataset.

Optimiser: AdamW, lr 1e-4, wd 1e-4, grad clip 1.0, batch 4, 10 epochs (2 warm-up, cosine), optional EMA.

## 7. Inference

1. `score = sigmoid(event)^0.7 * sigmoid(IoU)^0.3` per point (`iou_power: 0.3`), keep points above 0.05 (at least one), top 500.
2. Segment of a point = `[p - off0 * stride, p + off1 * stride]`; drop those shorter than `duration_thresh`.
3. Class-agnostic soft-NMS (IoU 0.7, sigma 0.5), at most 100 events; convert steps to seconds.
4. `q = SegmentContext(...)` for each kept event.
5. **Caption, retrieval:** cosine of `q` with the pool in the event space, top 20, softmax with the learned scale as weights, choose the
   candidate with the highest weighted agreement with the others in InternVideo2 text space (minimum Bayes risk). The result keeps 3 alternatives.
6. **Caption, generation (optional, `caption_mode='generate'`):** `tools/capgen/` GPT-2 reads `q` and K level-0 feature tokens as a prefix
   (optionally the retrieved captions as text) and writes the caption.
7. **Ground:** for a sentence, `sigmoid(ground logit)^0.7 * sigmoid(IoU)^0.3` over all points, best point, its boundary-head segment.

## 8. Metrics

R@0.5 / R@0.7 (segmentation), CIDEr / METEOR (captions), `txt_sim` (cosine of chosen and GT caption in InternVideo2 text space),
`G R1@0.5` / `G R1@0.7` / `G mIoU` (each GT caption must find its own span). Seed noise: about ±1.5 R@0.5, ±2 CIDEr, ±0.003 txt_sim.
