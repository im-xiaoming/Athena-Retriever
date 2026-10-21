#!/bin/bash
# Kaggle side (notebook with GPU T4 x2, Internet on): this machine's two shares of the HT-Step pipeline, one per GPU
# (download + k400_f16 extraction + upload to HF). Set the secret HF_TOKEN (Add-ons > Secrets) first; its account must
# have accepted the licence of OpenGVLab/InternVideo2-Stage1-1B-224p-K400.
# Run in a notebook cell:
#   !git clone -q -b pipeline-k400 https://github.com/<user>/UniAV /kaggle/working/UniAV
#   import os; from kaggle_secrets import UserSecretsClient; os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
#   !bash /kaggle/working/UniAV/tools/kaggle_work.sh
set -e
cd /kaggle/working/UniAV
pip install -q "huggingface_hub==0.25.2" yt-dlp timm einops >/dev/null 2>&1
command -v deno >/dev/null || curl -fsSL https://deno.land/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.deno/bin:$PATH
[ -d InternVideo ] || git clone -q --depth 1 https://github.com/OpenGVLab/InternVideo
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
for repo, f in (('OpenGVLab/InternVideo2-Stage1-1B-224p-K400', '1B_ft_k710_ft_k400_f16.pth'),
                ('OpenGVLab/InternVideo2-Stage2-6B-Audio', 'audio_6b.pth')):
    print(hf_hub_download(repo, f, local_dir='ckpt/internvideo2', token=os.environ['HF_TOKEN']), flush=True)
PY
for g in 0 1; do   # GPU g takes shards_part1_kaggle<g>.csv; each share has its own folders and logs
  CUDA_VISIBLE_DEVICES=$g SMALL_DISK=1 VIDEOS=/kaggle/working/videos$g OUT=/kaggle/working/feats$g \
    LOGS=/kaggle/working/logs PY=python bash tools/pipeline/run.sh tools/pipeline/shards_part1_kaggle$g.csv
done
