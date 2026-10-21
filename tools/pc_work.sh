#!/bin/bash
# PC worker: extract features for the remaining HT-Step shards (claims keep it from colliding with Colab).
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PATH=$HOME/.local/ffbin:$PATH
/home/minh/uniav-api-env/bin/python -u tools/htstep_hub.py work --name pc --tmp /tmp/pc_work_htstep \
  --repo InternVideo --video-ckpt ckpt/internvideo2/InternVideo2-stage2_1b-224p-f4.pt \
  --audio-ckpt ckpt/internvideo2/audio_6b.pth --workers 2
