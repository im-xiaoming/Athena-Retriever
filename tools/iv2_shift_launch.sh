#!/bin/bash
# Start the half-second shifted InternVideo2 extraction once Drive is mounted on the session, then
# hand over to tools/iv2_finish.sh (check, Drive + HF copies, local copy, stop the session).
#   setsid nohup tools/iv2_shift_launch.sh <session> > /dev/null 2>&1 < /dev/null &
SESSION=${1:-iv2shift}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG=$ROOT/logs/iv2_shift_launch.log
C=~/.local/bin/colab
say() { echo "$(date '+%m-%d %H:%M:%S')  $*" >> "$LOG"; }
rexec() {
  printf 'import subprocess\nprint(subprocess.run(r"""%s""", shell=True, capture_output=True, text=True).stdout)\n' "$1" \
    | timeout 150 $C exec -s "$SESSION" 2>/dev/null
}
say "waiting for Drive on $SESSION"
until rexec 'ls "/content/drive/MyDrive/code KL/Omni/data/YouCookII/videos" | head -1' | grep -q mp4; do sleep 60; done
say "Drive mounted, starting /content/iv2_full.sh (shift 0.5)"
rexec 'cd /content && nohup setsid bash /content/iv2_full.sh > /dev/null 2>&1 < /dev/null &' > /dev/null
sleep 30
cd "$ROOT" && IV2_NAME=iv2_feats_shift exec tools/iv2_finish.sh "$SESSION"
