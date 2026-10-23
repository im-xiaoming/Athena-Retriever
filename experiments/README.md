# Experiments

Where every training result and comparison is kept. A run name leads to its record.

| Path | What | Written |
|---|---|---|
| `runs/<host>-<run>.json` | one record per training run: command, commit, full config, every epoch's metrics, best checkpoints, final evaluation | by `train.py` |
| `RESULTS.md` | one row per run record, sorted by CIDEr | `python tools/summarize_runs.py` |
| `NOTES.md` | what each change bought, lessons (rounds up to 2026-10-08) | by hand |
| `plots/` | training curves of the 2026-10-04 rounds | `tools/plot_runs.py` |

The reports, benchmark scores and analysis scripts of the earlier rounds (YouCook2 / COIN / ActivityNet) are in the git history.
Notes of the first rounds: `NOTES.md`.

## Conventions

- Run names: `<data>_<change>` (e.g. `all_b16` = all training sets, batch 16). The host prefix of a record tells
  where it ran (`DESKTOP-8PSQBN9` = the PC, a hex id = a Colab VM).
- Compare runs on the best-epoch numbers and within seed noise: R@0.5 ±1.5, CIDEr ±2, txt_sim ±0.003.
- Checkpoints are not in git: `ckpt/<run>/best_seg.pth.tar` (best R@0.5) and `best_cap.pth.tar` (best CIDEr); the
  report of a run says where they are kept.
