# UniAV for YouCook2: event segmentation + captioning

Finds the steps of a cooking video and describes each one. Built on the backbone of
[UniAV](https://arxiv.org/abs/2404.03179) (audio-visual ConvTransformer, itself based on ActionFormer),
reduced to a single task:

- **Input**: InternVideo2-1B video features + BEATs audio features, one row per second
  (`tools/extract_internvideo2.py`).
- **Segmentation**: class-agnostic event head, boundary head with an IoU-quality branch, soft-NMS.
- **Captioning**: each event's vector (span mean + video context) is matched against the 8218 train
  captions in a text space (ONE-PEACE by default, or InternVideo2's own text space) and a caption is
  picked by consensus; an optional GPT-2 generator writes captions instead (`tools/capgen/`).
- **Teacher**: OmniRetriever-7B embeddings of every GT clip and caption guide the event vectors
  during training (off at inference).

Architecture: `docs/UniAV_new.drawio.svg`, walkthrough with an example: `docs/UniAV_new_walkthrough.md`.

| YouCook2 val (394 videos) | R@0.5 | R@0.7 | ret_sim | CIDEr | METEOR |
|---|---|---|---|---|---|
| `iv2` (API model) | 54.3 | 31.2 | 0.754 | 87.9 | 15.56 |
| + GPT-2 captions on the same events | | | | 95.1 | 15.42 |

All runs, configs and lessons: `experiments/RESULTS.md`, `experiments/NOTES.md`.

## Layout

| Path | What |
|---|---|
| `train_event.py` | training and evaluation; writes `experiments/runs/<host>-<run>.json` per run |
| `configs/youcook2_event.yaml` | the config; every option can be overridden with `--set key=value` |
| `libs/datasets/youcook2_cap.py` | dataset: features, caption pool, teacher vectors |
| `libs/modeling/event_archs.py` | the model and its losses |
| `libs/modeling/multimodal_backbones.py`, `blocks.py` | audio-visual ConvTransformer backbone |
| `uniav_api/` | inference API (video or stored features -> events with captions, search); see its README |
| `tools/` | feature extraction, caption vectors, teacher vectors, API export, plots, caption generator |
| `docs/` | architecture, handoffs between sessions, data locations |

## Setup

Training: Python 3.8+, PyTorch 1.11+ (also runs on 2.x), numpy, pyyaml, pandas, h5py, tensorboard,
pycocoevalcap (METEOR needs `java`). The 1-D NMS is a C++ extension; build it after every PyTorch change:

```bash
cd libs/utils && python setup.py install --user && cd ../..
```

Data (private HF dataset `nguyenminh04/uniav-youcook2-data`, see `docs/HANDOFF_KAGGLE.md`):
`iv2_feats/` -> `data/youcookii/iv2_feats/` (1500 `.npz`), `teacher/omni_emb_full.npz` ->
`data/youcookii/`. Annotations and caption vectors (`caption_emb.npz`, `caption_emb_iv2.npz`) are in git.

## Train and evaluate

```bash
python train_event.py configs/youcook2_event.yaml --output iv2 --note 'baseline'
python train_event.py configs/youcook2_event.yaml --output iv2_txt --set dataset.caption_space=iv2
python train_event.py configs/youcook2_event.yaml --eval ckpt/iv2/best_cap.pth.tar
python tools/summarize_runs.py && python tools/plot_runs.py
python tools/export_api_ckpt.py iv2          # -> ckpt/api/uniav_iv2.pth for uniav_api
```

About 20 minutes per run on an RTX 3060 (2 warm-up + 10 cosine epochs). Seed noise is about
±1.5 R@0.5, ±0.003 ret_sim, ±2 CIDEr: compare variants over two seeds (`--set init_rand_seed=2024`).

## Credits

UniAV (Geng et al., 2024, [code](https://github.com/ttgeng233/UniAV)), ActionFormer, UnAV,
InternVideo2, BEATs, ONE-PEACE, OmniRetriever.
