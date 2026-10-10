#!/bin/bash
# PC worker: extract features for the remaining shards of the benchmarks (dcase, unav100) and anet/htstep (claims keep it from colliding with Colab).
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PATH=$HOME/.local/ffbin:$PATH
for ds in dcase unav100 anet htstep; do
  /home/minh/uniav-api-env/bin/python -u tools/coin_hub.py work --dataset $ds --name pc --tmp /tmp/pc_work_$ds \
    --repo InternVideo --video-ckpt ckpt/internvideo2/InternVideo2-stage2_1b-224p-f4.pt \
    --audio-ckpt ckpt/internvideo2/audio_6b.pth --workers 2
done
