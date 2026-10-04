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
| Teacher at 100% coverage (`omni100`, `omni100_seed2`) | same as 65%: ret_sim 0.754–0.755, top-1 0.736–0.739 | yes |
| Teacher weights 0.5/0.5 (`omni100_w05`) or av target only (`omni100_avonly`) | within noise of the default 0.2/0.2 | default |
| Span-mean vector instead of single anchor point | no difference | kept, needed by the teacher |
| InternVideo2 v768 + BEATs a768 instead of ONE-PEACE (`iv2`, `iv2_seed2`; same teacher, full data) | R@0.5 47.4–49.1 → 52.8–54.3, R@0.7 +5; ret_sim, top-1 within noise; CIDEr +2 | yes, API model |
| Add the text-aligned v512 to v768 (`iv2_v512`) | between ONE-PEACE and v768 alone on R@0.5 (52.9); captions same | no |
| InternVideo2 + ONE-PEACE concatenated (`iv2op`) | R@0.5 48.8: ONE-PEACE's large raw norms (~27) swamp the L2-normalised InternVideo2 rows; stuck near 35 until epoch 6 | no, ONE-PEACE dropped |
| Teacher score added at caption choice (vs caption space only) | ret_sim changes by <= 0.003 in every run | API uses caption space only |
| Inference sweep on `iv2` (same checkpoint): `iou_power` 0.2/0.3/0.5 x soft-NMS sigma 0.5/0.9 | best `iou_power 0.5`: R@0.5 54.27 → 54.97, R@0.7 +0.6; sigma 0.9: R@0.7 +0.7 | not applied yet (API min_score is calibrated for 0.3) |
| No teacher with InternVideo2 (`iv2_noteach`) | R@0.5 51.4 (iv2 seeds 52.8–54.3), captions same | keep teacher (cheap, slightly better segmentation) |
| Dropout 0.1 + drop-path 0.2 (`iv2_reg`) | within noise everywhere | no |
| Embedding / span loss weights 0.2 → 0.4 (`iv2_emb04`) | within noise everywhere | no |
| 8 epochs instead of 10 (`iv2_ep8`, `iv2_ep8_seed2`) | same as 10 within noise (R@0.5 52.9 vs 53.5, ret_sim 0.758 vs 0.756), 20% faster | optional |
| 8 epochs + dropout (`iv2_ep8_reg`) | R@0.5 54.4, single run, inside iv2's seed range | no |

| max_seq_len 512 with the 1 row/s features (`iv2_len512`) | segmentation same (R@0.5 53.0); captions best single run so far: ret_sim 0.761, CIDEr 91.1, METEOR 16.14 | promising, needs a second seed |
| OmniRetriever text as the caption space instead of ONE-PEACE (`txt_omni`) | all within noise (ret_sim 0.756, CIDEr 87.8) | ONE-PEACE text can be dropped without loss |
| Teacher target = clip audio+video + its caption, T+V+A (`teach_tva`) | captions up (ret_sim 0.760, CIDEr 92.2, METEOR 16.09), R@0.5 51.8 (-1.7) | promising for captions, single run |
| Caption generator: GPT-2 with a segment prefix (`tools/capgen`, on the iv2 model) | val GT spans: CIDEr 97.6 vs 88.9 retrieval, METEOR 15.72 vs 15.13, BLEU-4 8.4 vs 7.0; predicted segments: CIDEr 95.1 vs 86.0 | yes, best captioning so far |
| Generated vs retrieved, ret_sim (ONE-PEACE text cosine to GT, `score_retsim.py`) | generated 0.727 vs retrieval 0.753 on both GT and predicted segments. ret_sim favours retrieval: the pick is made in the same ONE-PEACE space the metric uses; CIDEr/METEOR do not depend on it. Lengths match (8.2 vs 8.4 words, GT 9.0); generation is more varied (1958 vs 1273 distinct captions on 2253 segments) | report both; generation wins on n-gram metrics only |
| Caption generator with the 5 retrieved captions in the prompt (`rag5`, `--rag 5`) | GT spans CIDEr 99.0 (prefix 97.6), predicted segments 91.2 (prefix 95.1), ret_sim 0.730 (+0.003); train loss 0.96 vs prefix, overfits the candidates; 3x slower to train | no, API keeps the prefix generator |

Conclusion of the variants (2026-10-04 afternoon): the InternVideo2 model is on a plateau for
these hyper-parameters; every change lands inside the seed noise. Further gains need a bigger
change (see docs/HANDOFF.md, next steps).
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

## Round 2 conclusion (2026-10-04)

Across 5 teacher runs vs 2 no-teacher runs, the teacher is a consistent captioning
gain: ret_sim 0.753–0.758 vs 0.748–0.751, top-1 0.734–0.740 vs 0.722–0.729, METEOR
15.1–15.4 vs 14.4–14.6. The weighting does not matter. Segmentation is unchanged
(R@0.5 47–50 everywhere), so the teacher only helps the representation of a span,
and only a little: ret_sim is still far from the 0.896 ceiling.
Scoring with the teacher space alone (ret_sim[mbr_omni] 0.746) is worse than the
ONE-PEACE space, so the gain comes from better segment vectors, not from retrieval
in the teacher's space.

## Planned combinations

1. Teacher at 100% coverage, default weights (the baseline for everything after).
2. Teacher loss weights 0.5 / 0.5, and av-only teacher (text weight 0): which target matters.
3. Second seed of the best teacher run, to confirm it beats noise.
4. Then add text-aligned SigLIP features on top of the best teacher setting.
