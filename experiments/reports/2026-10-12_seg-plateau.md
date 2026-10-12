# 2026-10-12: trying to raise YouCook2 R@0.5 above the 54-55 plateau

**Question:** R@0.5 stays at 54-55 whatever the training data (2026-10-11 round). The diagnosis in
`2026-10-11_all-data.md` ("Why R@0.5 stays at 54-55") shows the proposals are there (92.3 % of GT steps have a
proposal with IoU >= 0.5 among the 100 kept per video) but the K kept for scoring are not the right ones (54.0 %).
Can training changes or a re-ranking step pick better K? Only R@0.5 counts here (the user's metric).

## Summary

Nothing tried moves R@0.5 beyond seed noise (±1.5). Best results, all within noise of `all_b16` (53.98):

| Attempt | Kind | R@0.5 | vs `all_b16` |
|---|---|---|---|
| 1. `ft_yc2`: fine-tune `all_b16` best_seg on YouCook2 only | training | 54.07 (epoch 1, then down to 52.2) | +0.1 |
| 2. `ft_other`: as 1 + `loss_weight_other` 0.2 (background steps pulled to the text "other") | training | 54.17 (epoch 1, then down to 52.3) | +0.2 |
| 2b. rerank `ft_other` proposals by p(other) | inference | -0.27 ± 0.38 (2-fold) | 0 |
| 3. `all_src`: `all_b16` + `model.source_token=4` (a learned vector per dataset for the event and boundary heads) | training | 53.08 (epoch 8, stopped at 9) | -0.9 |
| 4. pick K non-overlapping segments (YouCook2 steps overlap in 0.5 % of cases) | inference | -0.06 ± 0.25 (2-fold) | 0 |
| 5. proposal re-ranker (transformer over the 100 proposals, overlap-aware attention, one-to-one targets) | trained post-hoc | best -0.30 ± 0.29 (2-fold on val), 52.99-54.34 (trained on train) | 0 |

Earlier the same day: reranking by caption-pool match and `iou_power` / soft-NMS sigma / length prior sweeps, also
no gain (see the previous report).

Conclusion: with the current features, the model's own score is already the best ranking that can be learned from
YouCook2's 1,106 training videos; the score, the segment vector `q`, the position and the overlaps between
proposals carry no extra information about which proposals are the annotated steps. In the whole history of the
project only two changes moved R@0.5: better input features (ONE-PEACE -> InternVideo2 + BEATs, +5) and the
IoU-quality ranking branch (+1). The next lever is the input: denser (2 fps) or stronger visual features, mainly
for the short steps (R@0.5 18 % for steps under ~6 s).

## Update: ensembling checkpoints does move R@0.5

No feature re-extraction (too expensive for the user). Five `best_seg` checkpoints already on the PC, all on the
same 504-point grid: E1 (`ckpt/api/athena.pth`, YouCook2 + EMA), R1 (`athena_coin.pth`, + COIN), `all_b16`,
`ft_yc2`, `ft_other`. Per point, the event and IoU probabilities and the boundary offsets are averaged over the
members, then the usual score, soft-NMS and top K. Fixed before looking at the numbers: all five members, voting
threshold 0.75 (UniAV's default).

| Model | R@0.5 | R@0.7 |
|---|---|---|
| E1 / R1 / `all_b16` / `ft_yc2` / `ft_other` alone | 55.39 / 53.55 / 53.98 / 54.07 / 54.14 | 31.97 / 31.08 / 32.04 / 32.70 / 32.00 |
| **ensemble of the 5** | **56.12** | **34.18** |
| ensemble of the 5 + segment voting (UniAV / ActionFormer `seg_voting`, IoU 0.75) | 56.15 | 34.71 |
| `all_b16` + segment voting | 53.68 | 32.86 |

Paired bootstrap over videos (2,000 resamples): ensemble - `all_b16` = +2.11 R@0.5 (95 % CI +0.95 to +3.28);
ensemble - E1 = +0.68 (CI -0.61 to +1.94, not significant). Pairs of differently trained models gain most
(E1 + `all_b16` 56.71, exploratory, picked on val), pairs of a run and its own fine-tune least (`all_b16` + `ft_yc2`
55.03): the gain comes from diversity between members. Segment voting only helps R@0.7.

Scripts: `ens.py` (runs the members, stores per-point outputs), `ens_eval.py`, `ens_boot.py`.

## Runs

| Run | Record | Code | Setup | Checkpoints |
|---|---|---|---|---|
| `ft_yc2` | `runs/DESKTOP-8PSQBN9-ft_yc2.json` | `aug-reg` 38db8cd + the uncommitted `--init` (= `seg-plateau` 90c3e83) | PC RTX 3060; `--init ckpt/all_b16/best_seg.pth.tar --epochs 4`, lr 5e-5, clip_proj_lr 5e-4, 1 warm-up epoch, YouCook2 only, batch 4 | local `ckpt/ft_yc2` |
| `ft_other` | `runs/DESKTOP-8PSQBN9-ft_other.json` | `seg-plateau` 0efc374 | as `ft_yc2` + `model.train_cfg.loss_weight_other=0.2` | local `ckpt/ft_other` |
| `all_src` | `runs/9f0e96df7205-all_src.json` | `seg-plateau` 90c3e83 | Colab A100; `all_b16` command + `model.source_token=4`; stopped by hand after epoch 9 (R@0.5 flat since epoch 3), no final evaluation | lost with the Colab session |

| Epoch | `ft_yc2` R@0.5 | `ft_other` R@0.5 | `all_src` R@0.5 | `all_b16` R@0.5 |
|---|---|---|---|---|
| 1 | **54.07** | **54.17** | 48.43 | 45.89 |
| 2 | 53.48 | 53.84 | 49.75 | 50.87 |
| 3 | 52.49 | 53.12 | 52.19 | 48.14 |
| 4 | 52.36 | 52.06 | 52.29 | 53.91 |
| 5 | 52.16 | 52.29 | 52.95 | 53.42 |
| 6 | | | 53.05 | 53.74 |
| 7 | | | 51.86 | 53.71 |
| 8 | | | **53.08** | **53.98** |
| 9 | | | 51.57 | 51.83 |

The fine-tunes improve captions (CIDEr 79.4 -> 89.8 for `ft_yc2`, txt_sim 0.471 -> 0.510) while R@0.5 falls from
the first epoch: YouCook2 alone overfits the segmentation. `all_src` helps COIN unseen accuracy at equal epochs
(12.2 vs 10.6 at epoch 6) but not YouCook2.

## Inference-only and post-hoc attempts (checkpoint `all_b16` best_seg, YouCook2 val)

Each parameter is chosen on one half of val and scored on the other half (random video splits), so the gains are
not picked on the numbers they report.

- **Non-overlapping K** (UniAV handles overlapping events with class-aware regression and per-class NMS; here the
  opposite applies, YouCook2 steps almost never overlap: 0.7 % of train steps, 0.5 % of val steps): greedy pick in
  score order, skip a segment overlapping a chosen one by more than a fraction f of the shorter. f = 1 (off) 53.98,
  0.5: 52.59, 0: 49.42. The current soft-NMS already leaves only 0.3 % duplicate slots.
- **p(other) of `ft_other`**: separates background at the point level (mean 0.82 on IoU < 0.3 proposals vs 0.50 on
  hits) but adds nothing to the event score: best lambda gives -0.27 ± 0.38. On the span vector `q` p(other) is
  ~0 everywhere.
- **Proposal re-ranker** (`reranker.py`): per video, the 100 proposals as tokens (log score, rank, start, end, centre,
  length, duration, optionally `q` and the point vector); 3 transformer layers, d 256, 4 heads; attention bias
  from pairwise IoU, containment, centre distance and length ratio. Output = base log score + a zero-initialised
  correction. Targets: one-to-one (each GT step labels only its best proposal with IoU >= 0.5, as in DETR; ranking
  by these targets gives R@0.5 92.3). Listwise loss.

  | Variant | 2-fold on val (gain) | trained on YouCook2 train, scored on val |
  |---|---|---|
  | target = max IoU, BCE | -3.94 ± 0.53 | 49.49 |
  | target = all hits, listwise, residual | -3.49 ± 1.04 | - |
  | one-to-one, listwise, with `q` | -1.30 ± 0.65 (duplicates 0.3 % -> 2 %) | 52.69 (54.01 at epoch 10) |
  | one-to-one, listwise, without `q` | **-0.30 ± 0.29** | 52.99 (54.34 at epoch 10) |

  Train proposals are easier than val ones (base R@0.5 63.5 vs 54.0 on the same checkpoint), and the 197 videos of a
  val half are too few: the re-ranker overfits whatever it is given, most of all `q`.

## Reproduce

Scripts: `scripts/2026-10-12_seg-plateau/` (run from the repo root with the training env; `<d>` is a scratch folder).

```bash
python experiments/scripts/2026-10-12_seg-plateau/dump_props.py ckpt/all_b16/best_seg.pth.tar experiments/runs/b8a86a5e6ef7-all_b16.json val <d>/props_val.pkl
python experiments/scripts/2026-10-12_seg-plateau/dump_props.py ckpt/all_b16/best_seg.pth.tar experiments/runs/b8a86a5e6ef7-all_b16.json train <d>/props_train.pkl
USE_Q=0 python experiments/scripts/2026-10-12_seg-plateau/reranker.py <d> B list    # 2-fold on val; A = train on train
python experiments/scripts/2026-10-12_seg-plateau/dump_preds.py ckpt/all_b16/best_seg.pth.tar experiments/runs/b8a86a5e6ef7-all_b16.json <d>/all_b16_val.pkl
python experiments/scripts/2026-10-12_seg-plateau/nonoverlap.py <d>/all_b16_val.pkl   # also analyze.py, slots.py, rerank.py on this file
python experiments/scripts/2026-10-12_seg-plateau/dump_other.py ckpt/ft_other/best_seg.pth.tar experiments/runs/DESKTOP-8PSQBN9-ft_other.json val <d>/props_other_val.pkl
python experiments/scripts/2026-10-12_seg-plateau/rerank_other.py <d>/props_other_val.pkl
```

## Next

- More diverse ensemble members at no feature cost: E1-recipe runs with other seeds on the PC (~20 min each), then
  the ensemble of all of them.

- Input features: InternVideo2 at 2 fps (re-extraction on Colab, ~29 video-s/s on an A100) with `max_seq_len` 512,
  aimed at the short steps; or a second visual feature next to InternVideo2.
- New options kept in the code: `train.py --init`, `--patience`, `model.source_token`.
