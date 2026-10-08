# Data pipeline: how videos become features on Hugging Face

Videos of every dataset go from YouTube to frozen InternVideo2 + BEATs features through Hugging Face (HF)
dataset repos, using any number of machines at once. Nothing here trains a model.

## Datasets and repos

| Dataset | Videos repo | Features repo (also claims, plan, annotations) | Shards |
|---|---|---|---|
| COIN (`--dataset coin`, default) | `nguyenminh04/coin-videos` | `nguyenminh04/coin-data` | 0000-0015, 1000-1099, 2000-2001; 5,180 videos with features; `exclude.txt` lists 3 ids shared with YouCook2 val |
| ActivityNet (`--dataset anet`) | `nguyenminh04/anet-videos` | `nguyenminh04/anet-feats` | 35 chunks, 1000-1034 |
| HT-Step (`--dataset htstep`) | `nguyenminh04/htstep-videos` | `nguyenminh04/htstep-feats` | 11 chunks, 1000-1010 (1,005 videos: val_seen + about 900 train) |
| YouCook2 | `nguyenminh04/uniav-youcook2-data` | same repo (`iv2_feats/`, teacher vectors) | 1,500 `.npz` |

Dataset cards: `docs/hf_cards/`. Repos are public. Tokens (`HF_TOKEN`, ...) live in `.env`; never print or commit it.

## Flow

```
plan ──► fetch (download) ──► videos repo ──► work (extract) ──► feats repo ──► training
 PC        any machine          *_videos_NNNN.tar     GPU machine      *_feats_NNNN.tar
```

All steps are subcommands of `tools/coin_hub.py` (`--dataset coin|anet|htstep`):

1. **`plan`** builds the list of video ids from the dataset annotations and splits them into chunks of 100
   (`dl/plan.json`; chunk numbers start at 1000). `plan --retry` makes `plan_retry.json` from the videos
   that failed with a retryable error.
2. **`fetch`** (any machine, no GPU) claims a chunk (`dl/claims/CCCC__<name>`), downloads it with yt-dlp
   (480p for COIN, 360p for ANET/HTStep; `--jobs`, `--cookies`, `--ffmpeg`, `--reverse`), tars it, uploads
   `<pfx>_videos_CCCC.tar` to the videos repo and writes `dl/done/CCCC.json` (ok / failed / retry ids).
   If YouTube blocks the IP ("not a bot", 429) it releases the chunk and waits `--block-wait` minutes.
   Cloud IPs (Colab, Kaggle) need no cookies; the PC needs login cookies.
3. **`work`** (GPU machine) claims a downloaded shard (`claims/NNNN__<name>`), extracts it with
   `tools/extract_internvideo2.py` and uploads `feats/<pfx>_feats_NNNN.tar` (one `<id>.npz` per video with
   `v768`, `v512`, `a768`, float16, one row per second, plus `failed.txt`). It exits when everything is
   done, and waits when nothing is free.
4. **`status`** prints downloaded / extracted counts. **`upload`** is the old PC-side packer used for COIN.

Rules that make machines interchangeable:
- A claim is a file named after the worker; the oldest claim wins; a claim older than 6 h is dead and can be
  taken over. Deleting a claim file frees the chunk by hand.
- Add or remove machines at any time. Each machine runs its own `fetch` and/or `work`.
- Checkpoints needed by `work`: `InternVideo2-stage2_1b-224p-f4.pt`, `audio_6b.pth`
  (`ckpt/internvideo2/`), and a clone of `OpenGVLab/InternVideo`.

## Machines

| Machine | Role | How |
|---|---|---|
| PC (WSL, RTX 3060 12 GB) | fetch + work | ffmpeg in `~/.local/ffbin`, worker env `uniav-api-env`; `logs/chain_htstep.sh`-style wrappers chain datasets |
| Colab free T4 | fetch + work | `tools/colab_dl_extract.sh` runs on the VM (installs yt-dlp, deno, checkpoints, starts both); `tools/colab_dl_launch.sh` retries session creation; `tools/colab_keepalive.sh` keeps it alive. Free sessions died after about 2 h; redeploy after deleting stale claims. `HOME=~/colabN` selects the Colab account |
| Kaggle (no GPU) | fetch only | `huggingface_hub==0.25.2` |

## Features in training

`libs/datasets/youcook2_cap.py` (YouCook2) and `libs/datasets/coin_cap.py` (COIN) read the features.
Design decisions for mixing datasets (one shared expert, caption pool per dataset, 1:1 source batches, GPT-2
caption generator with a source tag) and the COIN seen/unseen split are in `docs/PLAN_openvocab.md`.
Evaluation always includes YouCook2 val (394 videos); seed noise is about ±1.5 R@0.5, ±2 CIDEr.

## Other docs

- `ARCHITECTURE.md`: model architecture, losses, inference.
- `PLAN_openvocab.md`: open-vocabulary plan (OV-AVEL ideas).
- `../experiments/RESULTS.md`, `../experiments/NOTES.md`: runs and lessons. `../athena/README.md`: inference API.
