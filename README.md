# UniAV for YouCook2: event segmentation + captioning

Finds the steps of a cooking video and describes each one. Built on the backbone of
[UniAV](https://arxiv.org/abs/2404.03179) (audio-visual ConvTransformer, itself based on ActionFormer),
reduced to a single task:

- **Input**: InternVideo2-1B video features + BEATs audio features, one row per second
  (`tools/extract_internvideo2.py`).
- **Segmentation**: class-agnostic event head, boundary head with an IoU-quality branch, soft-NMS.
- **Captioning**: each event's vector (span mean + video context) is matched against the 8218 train
  captions in InternVideo2's own text space (the space of its v512 video projection) and a caption is
  picked by consensus; a GPT-2 generator writes captions instead (`tools/capgen/`).
- **Grounding**: a ground head finds the span a sentence describes (FiLM-conditioned on the sentence's
  InternVideo2 text vector, boundaries from the boundary head). Production takes a video, a video and
  a sentence, or a sentence alone (`uniav_api`, `uv.query`). Val metric: `G R1@0.5` / `G R1@0.7`
  (each GT caption must find its own span), `G mIoU`.
- **Teacher**: OmniRetriever-7B embeddings of every GT clip and caption guide the event vectors
  during training (off at inference).

Architecture: `docs/UniAV_new.drawio.svg`, walkthrough with an example: `docs/UniAV_new_walkthrough.md`.

| YouCook2 val (394 videos) | R@0.5 | R@0.7 | CIDEr | METEOR |
|---|---|---|---|---|
| `iv2` (ONE-PEACE caption space, before 2026-10-05) | 54.3 | 31.2 | 87.9 | 15.56 |
| + GPT-2 captions on the same events | | | 95.1 | 15.42 |
| `civ2_v512` (InternVideo2 caption space, 1 seed, Kaggle) | 53.1 | | 90.2 | |

Main metrics: R@0.5 / R@0.7 for segmentation, CIDEr / METEOR for captions. `txt_sim` (cosine of the
chosen and the GT caption in InternVideo2 text space) replaces the ONE-PEACE `ret_sim` of older runs.

All runs, configs and lessons: `experiments/RESULTS.md`, `experiments/NOTES.md`.

## Layout

| Path | What |
|---|---|
| `train_event.py` | training and evaluation; writes `experiments/runs/<host>-<run>.json` per run |
| `configs/youcook2_event.yaml` | the config; every option can be overridden with `--set key=value` |
| `libs/datasets/youcook2_cap.py` | dataset: features, caption pool, teacher vectors |
| `libs/modeling/event_archs.py` | the model and its losses (event, boundary, embedding and ground heads) |
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
`data/youcookii/`. Annotations and caption vectors (`caption_emb_iv2.npz`) are in git.

## Train and evaluate

```bash
python train_event.py configs/youcook2_event.yaml --output civ2 --note 'baseline'
python train_event.py configs/youcook2_event.yaml --output civ2_v512 --set 'dataset.iv2_video_keys=[v768,v512]'
python train_event.py configs/youcook2_event.yaml --eval ckpt/civ2/best_cap.pth.tar
python tools/summarize_runs.py && python tools/plot_runs.py
python tools/export_api_ckpt.py civ2_v512    # -> ckpt/api/uniav_civ2_v512.pth for uniav_api
```

About 20 minutes per run on an RTX 3060 (2 warm-up + 10 cosine epochs). Seed noise is about
±1.5 R@0.5, ±2 CIDEr, ±0.003 txt_sim: compare variants over two seeds (`--set init_rand_seed=2024`).

## Credits

UniAV (Geng et al., 2024, [code](https://github.com/ttgeng233/UniAV)), ActionFormer, UnAV,
InternVideo2, BEATs, OmniRetriever.
