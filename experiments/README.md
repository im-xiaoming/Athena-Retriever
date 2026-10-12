# Experiments

Where every training result and comparison is kept. Start from a report; a run name leads to its record.

| Path | What | Written |
|---|---|---|
| `reports/<date>_<topic>.md` | one round of training: question, runs, comparison with earlier runs, benchmark, conclusion | by hand |
| `runs/<host>-<run>.json` | one record per training run: command, commit, full config, every epoch's metrics, best checkpoints, final evaluation | by `train.py` |
| `RESULTS.md` | one row per run record, sorted by CIDEr | `python tools/summarize_runs.py` |
| `NOTES.md` | what each change bought, lessons (rounds up to 2026-10-08) | by hand |
| `benchmark/<date>_scores.json` | zero-shot DCASE / UnAV-100 scores against the original UniAV (`docs/BENCHMARK.md`) | `tools/eval_benchmark.py score --out` |
| `plots/` | training curves of the 2026-10-04 rounds | `tools/plot_runs.py` |
| `scripts/<date>_<topic>/` | one-off analysis scripts a report quotes, kept to reproduce it | by hand |

Rounds before 2026-10-11 are described in `NOTES.md` (2026-10-03/04, ONE-PEACE then InternVideo2 features) and
`docs/PLAN_openvocab.md` (2026-10-06/07, runs R0-R2, E1: grounding head, COIN, open vocabulary).

## Reports

| Date | Report | Runs | Main result |
|---|---|---|---|
| 2026-10-11 | [all training sets, regularisation + augmentation, text stream](reports/2026-10-11_all-data.md) | `all_b16`, `all_aug`, `all_text` | `all_b16`: YouCook2 same as YouCook2-only, COIN unseen Acc 12.5, UnAV-100 zero-shot mAP 23.0 (+6.3), overfits after epoch 8. `all_aug`: no overfitting, same segmentation (R@0.5 54.2) and COIN unseen (12.7), CIDEr 84.6 (-6). `all_text` (text as a third backbone stream): best run, G R1@0.5 49.1 (+3.4), COIN unseen Acc 14.2, R@0.5 54.6, CIDEr 88.5 |
| 2026-10-12 | [raising YouCook2 R@0.5 above 54-55](reports/2026-10-12_seg-plateau.md) | `ft_yc2`, `ft_other`, `all_src` | nothing moves R@0.5 beyond noise: YouCook2-only fine-tune, background loss, dataset token, non-overlap pick, a trained proposal re-ranker; an ensemble of 5 existing checkpoints does: R@0.5 56.1 (+2.1 over `all_b16`, +0.7 over E1) |
| 2026-10-12 | [architecture probes, then all datasets weighted equally](reports/2026-10-12_data-mix.md) | `e1_base`, `e1_cs15`, `e1_pself`, `e1_dense`, `e1_o2o`, `e1_tri`, `mix_eq`, `mix_noht` | centre sampling, OV-AVEL self-attention, 2 rows/s, one-to-one head, Trident head: YouCook2 R@0.5 within noise (Trident: R@0.7 +1-2.5). New metric R@0.5 avg over the 4 datasets: equal batch shares (`mix_eq`) 42.10 vs 41.08 for `all_b16` (one seed, COIN +3.9); HT-Step training adds nothing; ActivityNet is limited by short events, not overlaps |

## Conventions

- Run names: `<data>_<change>` (e.g. `all_b16` = all training sets, batch 16). The host prefix of a record tells
  where it ran (`DESKTOP-8PSQBN9` = the PC, a hex id = a Colab VM).
- Compare runs on the best-epoch numbers and within seed noise: R@0.5 ±1.5, CIDEr ±2, txt_sim ±0.003.
- Checkpoints are not in git: `ckpt/<run>/best_seg.pth.tar` (best R@0.5) and `best_cap.pth.tar` (best CIDEr); the
  report of a run says where they are kept.
