#!/bin/bash
# Colab side of the COIN extraction (docs/COIN_EXTRACT.md): set up once, then run one worker.
# Expects the HF token in /content/hf_token. Log: /content/coin_worker.log
set -e
cd /content
[ -d Athena-Retriever ] || git clone -q -b ov-refine https://github.com/im-xiaoming/Athena-Retriever.git Athena-Retriever
git -C Athena-Retriever pull -q
[ -d InternVideo ] || git clone -q --depth 1 https://github.com/OpenGVLab/InternVideo
export HF_TOKEN=$(cat /content/hf_token)
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
for repo, f in (('OpenGVLab/InternVideo2-Stage2_1B-224p-f4', 'InternVideo2-stage2_1b-224p-f4.pt'),
                ('OpenGVLab/InternVideo2-Stage2-6B-Audio', 'audio_6b.pth')):
    print(hf_hub_download(repo, f, local_dir='/content/iv2_ckpt', token=os.environ['HF_TOKEN']), flush=True)
PY
cd Athena-Retriever
exec python -u tools/coin_hub.py work --name ${WORKER_NAME:-colab-t4} --tmp /content/coin_work \
  --repo /content/InternVideo --video-ckpt /content/iv2_ckpt/InternVideo2-stage2_1b-224p-f4.pt \
  --audio-ckpt /content/iv2_ckpt/audio_6b.pth --workers 2
