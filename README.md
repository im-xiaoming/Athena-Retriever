# Event segmentation + captioning on YouCook2 (minimal version)

A short, read-first version of the model: find the steps of a cooking video and pick a caption for each.
Only the essentials are here. The full code, with every experiment option, is on the training branches.

## Files

| File | What it holds |
|---|---|
| `config.py` | every setting (paths, model size, loss weights, training) |
| `dataset.py` | YouCook2 videos: features resampled to 256 steps, GT steps, training targets of every pyramid point |
| `model.py` | backbone (UniAV, video + audio streams), the four heads, losses, inference (soft-NMS) |
| `train.py` | training loop with EMA weights, evaluation (R@0.3/0.5/0.7, mIoU, txt_sim, CIDEr, grounding) |

## Data

`data/youcookii/` already holds the annotations and the caption vectors (`caption_emb_iv2j.npz`, InternVideo2 text
vectors). The features are on HF `nguyenminh04/uniav-youcook2-data` (`iv2_feats/iv2_feats_*.tar`): extract every tar
into `data/youcookii/iv2_feats/`, one `<video_id>.npz` per video (InternVideo2 video `v768` + BEATs audio `a768`,
one row per second).

## Run

    pip install torch numpy pycocoevalcap
    python train.py

The checkpoint with the best R@0.5 is saved to `ckpt/minimal/best.pth.tar`.
