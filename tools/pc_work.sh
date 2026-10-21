#!/bin/bash
# PC worker: extract features for the remaining HT-Step shards (claims keep it from colliding with Colab).
# MODEL=k400_f16 (default, -> feats_k400/) or MODEL=stage2_f4 (the first features, -> feats/).
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PATH=$HOME/.local/ffbin:$PATH
MODEL=${MODEL:-k400_f16}
CKPT=ckpt/internvideo2/$([ "$MODEL" = k400_f16 ] && echo 1B_ft_k710_ft_k400_f16.pth || echo InternVideo2-stage2_1b-224p-f4.pt)
/home/minh/uniav-api-env/bin/python -u tools/htstep_hub.py work --name pc --model $MODEL --tmp /tmp/pc_work_htstep \
  --repo InternVideo --video-ckpt $CKPT --audio-ckpt ckpt/internvideo2/audio_6b.pth --workers 2
