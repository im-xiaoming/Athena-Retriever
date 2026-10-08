#!/bin/bash
# Colab side: download htstep videos (fetch) and extract features (work) at the same time.
# Expects /content/hf_token and /content/coin_hub.py (uploaded by colab_dl_launch.sh).
cd /content
export HF_TOKEN=$(cat /content/hf_token)
pip install -q yt-dlp huggingface_hub >/dev/null 2>&1
command -v deno >/dev/null || curl -fsSL https://deno.land/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.deno/bin:$PATH
[ -d InternVideo ] || git clone -q --depth 1 https://github.com/OpenGVLab/InternVideo
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
for repo, f in (('OpenGVLab/InternVideo2-Stage2_1B-224p-f4', 'InternVideo2-stage2_1b-224p-f4.pt'),
                ('OpenGVLab/InternVideo2-Stage2-6B-Audio', 'audio_6b.pth')):
    print(hf_hub_download(repo, f, local_dir='/content/iv2_ckpt', token=os.environ['HF_TOKEN']), flush=True)
PY
mkdir -p tools && cp /content/coin_hub.py tools/coin_hub.py
nohup python -u tools/coin_hub.py fetch --dataset htstep --name colab --tmp /content/fetch --reverse --jobs 3 > /content/fetch.log 2>&1 &
for ds in anet htstep; do
  python -u tools/coin_hub.py work --dataset $ds --name colab-t4 --tmp /content/work_$ds \
    --repo /content/InternVideo --video-ckpt /content/iv2_ckpt/InternVideo2-stage2_1b-224p-f4.pt \
    --audio-ckpt /content/iv2_ckpt/audio_6b.pth --workers 2 >> /content/work.log 2>&1
done
