#!/bin/bash
# PC side: keep asking for a free Colab T4; once granted, upload the files and start tools/colab_dl_extract.sh there.
# Usage: set -a; . ./.env; set +a; HOME=~/colab2 tools/colab_dl_launch.sh
cd "$(dirname "$0")/.."
C=/home/minh/.local/bin/colab
until $C new --gpu T4 -s t4 >/dev/null 2>&1; do echo "$(date +%T) no T4 yet"; sleep 300; done
echo "$(date +%T) T4 granted"
printf %s "$HF_TOKEN" > /tmp/hf_token_colab
$C upload -s t4 /tmp/hf_token_colab /content/hf_token && rm /tmp/hf_token_colab
$C upload -s t4 tools/coin_hub.py /content/coin_hub.py
$C upload -s t4 tools/colab_dl_extract.sh /content/colab_dl_extract.sh
$C upload -s t4 tools/extract_internvideo2.py /content/extract_internvideo2.py
echo "import subprocess; subprocess.Popen('nohup bash /content/colab_dl_extract.sh > /content/boot.log 2>&1 &', shell=True)" | $C exec -s t4
setsid nohup tools/colab_keepalive.sh t4 >/dev/null 2>&1 &
echo "$(date +%T) started"
