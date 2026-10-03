# Experiment notes

Lessons from each change and what to combine next. The results table is in
`RESULTS.md` (generated). Every run's config, commit and metrics are in `runs/`.

## How to read the numbers

- Seed-only noise, measured on two baseline runs: R@0.5 ±2, CIDEr ±5, ret_sim ±0.003,
  METEOR ±0.2. Treat smaller differences as noise.
- ret_sim and METEOR are the most stable captioning metrics. CIDEr swings 4–8 points
  between neighbouring epochs.
- ret_sim floor: 0.46 for a random caption. Ceiling: 0.896 if the best caption in the
  pool were always picked.
- Evaluation knows the number of GT segments per video, so it is easier than the
  PDVC/Vid2Seq protocol. Do not compare with those papers directly.

## What each change bought (on 3-epoch probes unless stated)

| Change | Effect | Keep |
|---|---|---|
| Consensus (MBR) caption pick over the top 20 | CIDEr 71 → 88, ret_sim +0.026 (same checkpoint) | yes |
| Embedding loss over the full 8218-caption pool, soft targets | top-1 ret_sim 0.694 → 0.711 | yes |
| IoU-quality branch for ranking, `iou_power: 0.3` | R@0.5 +1.0, R@0.7 +1.4 (same checkpoint) | yes |
| Embedding loss weights 1.0 → 0.2 | gives back 2 R@0.5 points lost to gradient clipping | yes |
| 10 epochs instead of 40 | 40 epochs overfit from epoch 7 (R@0.5 50.8 → 42.9 by epoch 12) | yes |
| Video context + position in the segment vector | CIDEr +2.5, within noise | yes (cheap) |
| OmniRetriever teacher, 65% coverage (`omni65`) | ret_sim +0.010, METEOR +1.0, top-1 +0.018; outside noise | yes |
| Span-mean vector instead of single anchor point | no difference | kept, needed by the teacher |
| Original UniAV checkpoint (`--pretrain`) | worse twice: R@0.5 −3.7, CIDEr −17 | no |
| `max_seq_len: 512` | same as 256 | no |
| Hard NMS / lower NMS IoU | worse than soft-NMS 0.7 | no |
| Strong regularisation (wd 0.05, dropout 0.1, crop 50–100%) | same as default | no |

## Facts that shape the next steps

- Captioning on GT spans is no better than on predicted spans (0.742 vs 0.748), so
  the bottleneck is the segment representation, not segmentation.
- OmniRetriever-7B zero-shot on 500 GT val clips: ret_sim 0.768, CIDEr 103.9. Our
  model on the same clips: 0.745, 96.2. Adding both scores did not beat the teacher
  alone, so the student adds nothing the teacher lacks.
- ONE-PEACE video features (Kinetics-400 fine-tuned) are not text aligned.
  Text-aligned frame features (SigLIP / CLIP at 1–2 fps from the uncut videos) are the
  next feature candidate.
- The model is data limited: it overfits by epoch 7. 290 annotated videos still have
  no features.

## Planned combinations

1. Teacher at 100% coverage, default weights (the baseline for everything after).
2. Teacher loss weights 0.5 / 0.5, and av-only teacher (text weight 0): which target matters.
3. Second seed of the best teacher run, to confirm it beats noise.
4. Then add text-aligned SigLIP features on top of the best teacher setting.
