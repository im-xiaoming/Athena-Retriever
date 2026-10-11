---
license: other
pretty_name: UniAV benchmark inputs (DCASE, UnAV-100)
tags:
- video
- audio
- temporal-localization
---

# UniAV benchmark inputs (private)

Test sets of [UniAV](https://github.com/ttgeng233/UniAV) (Geng et al., arXiv 2404.03179) as released by its
authors, so the original UniAV checkpoint (`nguyenminh04/uniav-original`) runs without the Baidu downloads.

| Path | Content |
|---|---|
| `annotations/dcase_all.json` | DESED `public_eval`, 564 clips of 10 s, 10 sound classes |
| `annotations/unav100_all.json` | UnAV-100 `test`, 2,167 videos, 100 audio-visual event classes |
| `onepeace/<set>/<set>_onepeace_NNNN.tar` | ONE-PEACE features: `<id>_one_peace_audio.npy`, `<id>_one_peace_video_finetune.npy` per video |

These features are UniAV's input only; our model uses InternVideo2 + BEATs features (`nguyenminh04/dcase-feats`,
`nguyenminh04/unav100-feats`). Code: `tools/benchmark_data.py`, `tools/eval_uniav.py`, `tools/eval_benchmark.py`
and `docs/BENCHMARK.md` in `im-xiaoming/Athena-Retriever`.
