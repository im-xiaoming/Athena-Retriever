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

Rounds before 2026-10-11 are described in `NOTES.md` (2026-10-03/04, ONE-PEACE then InternVideo2 features) and
`docs/PLAN_openvocab.md` (2026-10-06/07, runs R0-R2, E1: grounding head, COIN, open vocabulary).

## Reports

| Date | Report | Runs | Main result |
|---|---|---|---|
| 2026-10-11 | [all training sets, regularisation + augmentation](reports/2026-10-11_all-data.md) | `all_b16`, `all_aug` | YouCook2 same as YouCook2-only, COIN unseen Acc 12.5 (best), UnAV-100 zero-shot mAP 23.0 (+6.3); overfits after epoch 8 |

## Conventions

- Run names: `<data>_<change>` (e.g. `all_b16` = all training sets, batch 16). The host prefix of a record tells
  where it ran (`DESKTOP-8PSQBN9` = the PC, a hex id = a Colab VM).
- Compare runs on the best-epoch numbers and within seed noise: R@0.5 ±1.5, CIDEr ±2, txt_sim ±0.003.
- Checkpoints are not in git: `ckpt/<run>/best_seg.pth.tar` (best R@0.5) and `best_cap.pth.tar` (best CIDEr); the
  report of a run says where they are kept.
