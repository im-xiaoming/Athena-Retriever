#!/bin/bash
# Kaggle side (notebook with GPU T4 x2, Internet on): extract features for the benchmark shards, one worker per GPU.
# Claims on HF keep it from colliding with the PC and Colab. Set the secret HF_TOKEN (Add-ons > Secrets) first.
# Run in a notebook cell:
#   !git clone -q https://github.com/<user>/UniAV /kaggle/working/UniAV   # or upload tools/coin_hub.py + extract_internvideo2.py
#   import os; from kaggle_secrets import UserSecretsClient; os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
#   !bash /kaggle/working/UniAV/tools/kaggle_work.sh
set -e
W=/kaggle/working; cd $W
pip install -q "huggingface_hub==0.25.2" >/dev/null 2>&1
[ -d InternVideo ] || git clone -q --depth 1 https://github.com/OpenGVLab/InternVideo
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
for repo, f in (('OpenGVLab/InternVideo2-Stage2_1B-224p-f4', 'InternVideo2-stage2_1b-224p-f4.pt'),
                ('OpenGVLab/InternVideo2-Stage2-6B-Audio', 'audio_6b.pth')):
    print(hf_hub_download(repo, f, local_dir=os.environ.get('CKPT', '/kaggle/working/iv2_ckpt'), token=os.environ['HF_TOKEN']), flush=True)
PY
HUB=${HUB:-$W/UniAV/tools}
for g in 0 1; do
  (for ds in dcase unav100; do
     CUDA_VISIBLE_DEVICES=$g python -u $HUB/coin_hub.py work --dataset $ds --name kaggle-$g --tmp $W/work_${ds}_$g \
       --repo $W/InternVideo --video-ckpt $W/iv2_ckpt/InternVideo2-stage2_1b-224p-f4.pt \
       --audio-ckpt $W/iv2_ckpt/audio_6b.pth --workers 2
   done) > $W/work_$g.log 2>&1 &
done
wait
