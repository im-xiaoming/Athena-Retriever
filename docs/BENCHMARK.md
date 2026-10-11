# Benchmark: the original UniAV vs our model

Two test sets of the UniAV paper (Geng et al., arXiv 2404.03179), held out of all our training data:

| Set | Subset | Videos | Distinct GT events | With our features |
|---|---|---|---|---|
| DCASE / DESED (`benchmark/desed/dcase_all.json`) | `public_eval` | 564 clips of 10 s | 2,302 | 470 (the rest could not be downloaded) |
| UnAV-100 (`benchmark/unav100/unav100_all.json`) | `test` | 2,167 | 5,875 | 836 (only part of the set was extracted) |

ActivityNet 1.3 is not evaluated for now: the UniAV release has no ActivityNet features here.

## What is scored

UniAV names every event with a class; our model segments and captions, it has no classes. The metrics are
what both models do, **finding the events**, so class labels are dropped on both sides (`tools/eval_benchmark.py`):

- GT events with the same span are one event; UniAV's identical spans (one per class) are one prediction with
  the highest score.
- **AP@t / mAP**: ActivityNet average precision at tIoU t, UniAV's thresholds 0.1:0.9; mAP is their mean.
  Checked against UniAV's own evaluator with every label set to one class: same numbers.
- **R@t / mIoU**: as in `train.py`, each video keeps its k best predictions (k = its number of GT events).
- **common** rows: the videos both models have features for (the fair comparison); **all** rows: UniAV on
  the whole set, to compare with the paper.

UniAV is trained on the training splits of these datasets; our model never sees them (zero-shot).

## Files

| Where | What |
|---|---|
| `benchmark/UniAV/` | upstream code of github.com/ttgeng233/UniAV (MIT), unmodified; class prompts in `data/` |
| `nguyenminh04/uniav-benchmark` (HF, private) | `onepeace/<set>/*.tar`: ONE-PEACE features of the UniAV release (UniAV's input only), `annotations/` |
| `nguyenminh04/uniav-original` (HF, private) | `model_epoch_006.pth.tar`, the released UniAV checkpoint |
| `nguyenminh04/dcase-feats`, `unav100-feats` (HF) | InternVideo2 + BEATs features for our model (`tools/coin_hub.py work`) |

## Run

```bash
python tools/benchmark_data.py fetch            # ONE-PEACE features + checkpoint -> benchmark/, ckpt/uniav/; ours -> data/<set>/iv2_feats
python tools/eval_uniav.py                      # UniAV -> benchmark/preds/uniav_<set>.json (also prints its class-aware mAP)
python tools/eval_benchmark.py predict --ckpt ckpt/<run>/best_seg.pth.tar --run experiments/runs/<host>-<run>.json
python tools/eval_benchmark.py score            # table of every file in benchmark/preds/
```

Use the training env (torch 1.11 with `nms_1d_cpu`). `tools/eval_uniav.py` runs UniAV in its own process
(UniAV's `libs` package has the same name as ours), on one GPU without `torch.distributed`.
UniAV's own class-aware mAP from this setup: DCASE 59.79, UnAV-100 50.61.
