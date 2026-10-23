#!/bin/bash
# One machine's share: download (HF shards + YouTube) and extract in parallel, uploading each chunk's features to HF.
#   bash tools/pipeline/run.sh tools/pipeline/shards_part1_pc.csv [extra args for download_videos.py, e.g. --cookies a.txt]
# Env: VIDEOS (default ~/htstep_videos), OUT (default ~/iv2_k400), PY (python), SMALL_DISK=1 (Kaggle/Colab: delete videos
# after upload and keep at most 3 chunks ahead), LOGS (default logs_pipe). HF_TOKEN from the environment or .env.
# EX_ARGS: extra extract_features.py options (e.g. "--batch-size 32" on an A100).
# Stop with: pkill -f tools/pipeline/ ; rerun the same command to resume.
cd "$(dirname "$0")/../.."
CSV=$1; shift
PY=${PY:-python}; VIDEOS=${VIDEOS:-$HOME/htstep_videos}; OUT=${OUT:-$HOME/iv2_k400}; LOGS=${LOGS:-logs_pipe}
[ -f .env ] && { set -a; . ./.env; set +a; }
export PATH=$HOME/.local/ffbin:$PATH
mkdir -p "$LOGS"; TAG=$(basename "$CSV" .csv)
EXTRA_HF=""; EXTRA_EX=""
[ -n "$SMALL_DISK" ] && { EXTRA_HF="--max-ahead 3"; EXTRA_EX="--delete-videos"; }
nohup $PY -u tools/pipeline/download_hf.py --csv "$CSV" --out "$VIDEOS" --jobs 2 $EXTRA_HF > "$LOGS/$TAG.hf.log" 2>&1 &
[ -z "$NO_YT" ] && nohup $PY -u tools/pipeline/download_videos.py --csv "$CSV" --out "$VIDEOS" --jobs 4 "$@" > "$LOGS/$TAG.yt.log" 2>&1 &
nohup $PY -u tools/pipeline/extract_features.py --csv "$CSV" --videos "$VIDEOS" --out "$OUT" --upload --wait $EXTRA_EX $EX_ARGS > "$LOGS/$TAG.extract.log" 2>&1 &
echo "started $TAG; logs in $LOGS/$TAG.{hf,yt,extract}.log"
