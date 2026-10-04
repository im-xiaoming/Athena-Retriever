#!/bin/bash
# InternVideo2 features with every window moved by 0.5 s, for 2 rows per second once interleaved
# with the default extraction (tools/make_iv2_dense.py). Runs on a fresh Colab VM with Drive mounted.
# Upload as /content/iv2_full.sh (the name tools/iv2_finish.sh relaunches), with
# /content/extract_internvideo2.py, /content/iv2_finalize.py and /content/.hf_token, then
#   nohup setsid bash /content/iv2_full.sh > /dev/null 2>&1 &
# and watch it from the PC with IV2_NAME=iv2_feats_shift tools/iv2_finish.sh <session>.
# Safe to rerun: videos, weights and finished outputs are kept.
SRC="/content/drive/MyDrive/code KL/Omni/data/YouCookII/videos"
BACKUP=/content/drive/MyDrive/uniav_omni/iv2_feats_shift
VID=/content/yc2_videos; OUT=/content/iv2_feats; CK=/content/iv2_ckpt
mkdir -p "$VID" "$OUT" "$CK" "$BACKUP" /content/out
{
  echo "copying videos from Drive"
  cp -n "$SRC"/*.mp4 "$VID"/ 2>/dev/null
  echo "videos: $(ls "$VID" | wc -l)"
  [ -d /content/InternVideo ] || git clone -q --depth 1 https://github.com/OpenGVLab/InternVideo /content/InternVideo
  pip install -q einops timm
  python3 -c "
from huggingface_hub import hf_hub_download
tok = open('/content/.hf_token').read().strip()
for repo, name in (('OpenGVLab/InternVideo2-Stage2_1B-224p-f4', 'InternVideo2-stage2_1b-224p-f4.pt'),
                   ('OpenGVLab/InternVideo2-Stage2-6B-Audio', 'audio_6b.pth')):
    print(hf_hub_download(repo, name, local_dir='$CK', token=tok), flush=True)
"
} > /content/out/iv2_prep.log 2>&1
# backup of finished files every 10 minutes, and once more at the end
( while [ ! -f /content/out/iv2_full.done ]; do
    rsync -a --ignore-existing "$OUT"/ "$BACKUP"/; sleep 600
  done
  rsync -a --ignore-existing "$OUT"/ "$BACKUP"/
  echo BACKUP_DONE ) > /content/out/iv2_backup.log 2>&1 &
python3 /content/extract_internvideo2.py --videos "$VID" --out "$OUT" --repo /content/InternVideo --shift 0.5 \
  --video-ckpt "$CK/InternVideo2-stage2_1b-224p-f4.pt" --audio-ckpt "$CK/audio_6b.pth" > /content/out/iv2_full.log 2>&1
touch /content/out/iv2_full.done
