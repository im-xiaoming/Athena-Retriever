#!/bin/bash
# Extract InternVideo2 features for the YouCook2 videos that have none on Drive yet.
# Fresh Colab VM with Drive mounted (the user runs `colab drivemount`), then from the PC:
#   colab upload -s <s> tools/extract_internvideo2.py /content/extract_internvideo2.py
#   colab upload -s <s> tools/iv2_missing_colab.sh /content/iv2_missing_colab.sh
#   colab upload -s <s> <file with the HF token> /content/.hf_token
#   exec on the VM: nohup setsid bash /content/iv2_missing_colab.sh > /content/out/iv2_missing.log 2>&1 &
# New .npz files are copied to Drive; the last line of the log is ALL_DONE when finished.
set -e
SRC="/content/drive/MyDrive/code KL/Omni/data/YouCookII/videos"
DONE=/content/drive/MyDrive/uniav_omni/iv2_feats
VID=/content/yc2_videos; OUT=/content/iv2_feats; CK=/content/iv2_ckpt
mkdir -p "$VID" "$OUT" "$CK" /content/out

for f in "$SRC"/*.mp4; do
  v=$(basename "$f" .mp4)
  [ -f "$DONE/$v.npz" ] || cp -n "$f" "$VID/"
done
echo "videos without features: $(ls "$VID" | wc -l)"

[ -d /content/InternVideo ] || git clone -q --depth 1 https://github.com/OpenGVLab/InternVideo /content/InternVideo
pip install -q einops timm
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
tok = open('/content/.hf_token').read().strip()
for repo, name in (('OpenGVLab/InternVideo2-Stage2_1B-224p-f4', 'InternVideo2-stage2_1b-224p-f4.pt'),
                   ('OpenGVLab/InternVideo2-Stage2-6B-Audio', 'audio_6b.pth')):
    print(hf_hub_download(repo, name, local_dir='/content/iv2_ckpt', token=tok), flush=True)
EOF

python3 /content/extract_internvideo2.py --videos "$VID" --out "$OUT" --repo /content/InternVideo \
  --video-ckpt "$CK/InternVideo2-stage2_1b-224p-f4.pt" --audio-ckpt "$CK/audio_6b.pth"
cp -n "$OUT"/*.npz "$DONE/"
echo "on Drive now: $(ls "$DONE" | grep -c '\.npz$') files"
rm -f /content/.hf_token
echo ALL_DONE
