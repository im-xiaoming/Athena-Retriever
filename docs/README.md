# Data pipeline: how HT-Step videos become features on Hugging Face

HT-Step videos go from YouTube to frozen InternVideo2 + BEATs features through Hugging Face (HF) dataset repos,
using any number of machines at once. Nothing here trains a model.

## Repos

| Videos repo | Features repo (also claims, plans, annotations) | Shards |
|---|---|---|
| `nguyenminh04/htstep-videos` (`videos/htstep_videos_NNNN.tar`, `videos/manifest.json`) | `nguyenminh04/htstep-feats` (`feats/htstep_feats_NNNN.tar`) | chunks from 1000 (first plan: 1,005 videos, val_seen + about 900 train), retry chunks from 2000 and 4000 |

Tokens (`HF_TOKEN`, ...) live in `.env`; never print or commit it.

## Flow

```
plan ──► fetch (download) ──► videos repo ──► work (extract) ──► feats repo ──► htstep_data.py ──► training
 PC        any machine     htstep_videos_NNNN.tar   GPU machine   htstep_feats_NNNN.tar
```

All steps are subcommands of `tools/htstep_hub.py`:

1. **`plan`** picks the video ids from the HT-Step annotations (`datasets/htstep/data`: every `val_seen` video and
   `--n` random train videos, steps with a "Full Match" only) and splits them into chunks of 100 (`dl/plan.json`;
   chunk numbers start at `--first`, 1000), with their annotations in `anno/htstep.json`. `plan --retry` makes
   `dl/plan_retry.json` from the videos that failed with a retryable error.
2. **`fetch`** (any machine, no GPU) claims a chunk (`dl/claims/CCCC__<name>`), downloads it with yt-dlp
   (360p; `--jobs`, `--cookies`, `--ffmpeg`, `--reverse`, `--plan`), tars it, uploads `htstep_videos_CCCC.tar` to
   the videos repo and writes `dl/done/CCCC.json` (ok / failed / retry ids). If YouTube blocks the IP ("not a bot",
   429, or more "video unavailable" than downloads in a chunk) it releases the chunk and waits `--block-wait`
   minutes. Cloud IPs (Colab, Kaggle) need no cookies; the PC needs login cookies.
3. **`work`** (GPU machine) claims a downloaded shard (`claims/NNNN__<name>`), extracts it with
   `tools/extract_internvideo2.py` and uploads `feats/htstep_feats_NNNN.tar` (one `<id>.npz` per video with
   `v768`, `v512`, `a768`, float16, one row per second, plus `failed.txt`). It exits when everything is
   done, and waits when nothing is free.
4. **`status`** prints downloaded / extracted counts.

Rules that make machines interchangeable:
- A claim is a file named after the worker; the oldest claim wins; a claim older than `--stale` hours (6 for work,
  2 for fetch) is dead and can be taken over. Deleting a claim file frees the chunk by hand.
- Add or remove machines at any time. Each machine runs its own `fetch` and/or `work`.
- Checkpoints needed by `work`: `InternVideo2-stage2_1b-224p-f4.pt`, `audio_6b.pth`
  (`ckpt/internvideo2/`), and a clone of `OpenGVLab/InternVideo`.

## Machines

| Machine | Role | How |
|---|---|---|
| PC (WSL, RTX 3060 12 GB) | fetch + work | ffmpeg in `~/.local/ffbin`, worker env `uniav-api-env`; `tools/pc_work.sh` extracts |
| Colab T4 | fetch + work | `tools/colab_dl_extract.sh` runs on the VM (installs yt-dlp, deno, checkpoints, starts the worker); `tools/colab_dl_launch.sh` retries session creation; `tools/colab_keepalive.sh` keeps it alive; `tools/colab_watchdog.sh` relaunches a dead job. `HOME=~/colabN` selects the Colab account |
| Kaggle 2x T4 | work | `tools/kaggle_work.sh`, one worker per GPU |

## Features in training

`tools/htstep_data.py build --upload` (PC, GPU) turns `anno/htstep.json` into `anno/annotations.json`
(`{"database": {id: {"subset", "duration", "annotations": [{"segment", "sentence"}]}}}`) and InternVideo2 caption
vectors `text_feats/caption_emb_iv2j.npz` (centred on the mean of the training captions) in the features repo.
`tools/htstep_data.py fetch` puts features and both files in `data/htstep/`, which `libs/datasets/step_captions.py`
reads.

## Other docs

- `ARCHITECTURE.md`: model architecture, losses, inference.
- `../experiments/RESULTS.md`, `../experiments/NOTES.md`: runs and lessons.
