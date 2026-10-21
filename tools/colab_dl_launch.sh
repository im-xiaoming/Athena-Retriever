#!/bin/bash
# PC side: keep asking for a free Colab T4; once granted, upload the files and start tools/colab_dl_extract.sh there.
# Usage: set -a; . ./.env; set +a; HOME=~/colab2 tools/colab_dl_launch.sh
cd "$(dirname "$0")/.."
C=/home/minh/.local/bin/colab
$C new --gpu T4 -s t4 >/dev/null 2>&1 || { echo "no T4 for $HOME"; exit 1; }
echo "$(date +%T) T4 granted"
printf %s "$HF_TOKEN" > /tmp/hf_token_colab
$C upload -s t4 /tmp/hf_token_colab /content/hf_token && rm /tmp/hf_token_colab
$C upload -s t4 tools/colab_dl_extract.sh /content/colab_dl_extract.sh
$C upload -s t4 tools/pipeline/run.sh /content/run.sh
$C upload -s t4 tools/pipeline/extract_features.py /content/extract_features.py
$C upload -s t4 tools/pipeline/download_videos.py /content/download_videos.py
$C upload -s t4 tools/pipeline/download_hf.py /content/download_hf.py
$C upload -s t4 tools/pipeline/shards_part1_colab.csv /content/shards_part1_colab.csv
echo "import subprocess; subprocess.Popen('nohup bash /content/colab_dl_extract.sh > /content/boot.log 2>&1 &', shell=True)" | $C exec -s t4
setsid nohup tools/colab_keepalive.sh t4 >/dev/null 2>&1 &
echo "$(date +%T) started"
