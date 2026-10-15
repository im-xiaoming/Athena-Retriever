#!/bin/bash
# Colab / Kaggle side of tools/iv2_mid_hub.py: set up InternVideo2 and run one hub worker per GPU.
# Expects HF_TOKEN in the environment or in /content/hf_token. Usage:
#   NAME=colab-a100 nohup bash iv2_mid_colab.sh > /content/iv2_mid.log 2>&1 &
# Logs: $ROOT/iv2_mid_<name>.log. Safe to rerun.
BRANCH=${BRANCH:-iv2-mid}
ROOT=${ROOT:-/content}
NAME=${NAME:-colab}
cd "$ROOT"
[ -n "$HF_TOKEN" ] || export HF_TOKEN=$(cat "$ROOT/hf_token")
pip install -q einops timm >/dev/null 2>&1
# old images (Kaggle: 0.14) ignore HF_TOKEN; stay below 1.0 for older transformers
python -c "import huggingface_hub as h, sys; sys.exit(tuple(map(int, h.__version__.split('.')[:2])) < (0, 20))" || pip install -q "huggingface_hub>=0.20,<1.0" >/dev/null 2>&1
[ -d InternVideo ] || git clone -q --depth 1 https://github.com/OpenGVLab/InternVideo
mkdir -p tools
for f in extract_internvideo2.py iv2_mid_hub.py; do
  curl -fsSL "https://raw.githubusercontent.com/im-xiaoming/Athena-Retriever/$BRANCH/tools/$f" -o tools/$f
done
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('OpenGVLab/InternVideo2-Stage2_1B-224p-f4', 'InternVideo2-stage2_1b-224p-f4.pt', local_dir='$ROOT/iv2_ckpt'))"
NGPU=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NGPU - 1))); do
  n=$NAME; [ "$NGPU" -gt 1 ] && n=$NAME-g$g
  nohup python -u tools/iv2_mid_hub.py work --name $n --gpu $g --repo "$ROOT/InternVideo" \
    --video-ckpt "$ROOT/iv2_ckpt/InternVideo2-stage2_1b-224p-f4.pt" --tmp "$ROOT/iv2_mid_tmp_g$g" \
    ${ORDER:+--order $ORDER} > "$ROOT/iv2_mid_$n.log" 2>&1 &
  sleep 60   # stagger: the first claims of two workers should not collide
done
wait
