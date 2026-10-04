#!/bin/bash
# Keep a Colab CLI session alive: a tiny `colab exec` every 3 min, retried every 20 s on failure.
# Colab shuts a session ~25-30 min after the last exec from a client, even while jobs run.
#   nohup tools/colab_keepalive.sh [session] > /dev/null 2>&1 &      stop: touch logs/keepalive.stop
SESSION=${1:-omni}
DIR=$(cd "$(dirname "$0")/.." && pwd)/logs
mkdir -p "$DIR"
exec 9>"$DIR/keepalive.lock"; flock -n 9 || { echo "already running"; exit 0; }
C=~/.local/bin/colab
rm -f "$DIR/keepalive.stop"
while [ ! -f "$DIR/keepalive.stop" ]; do
  if echo "print('alive')" | timeout 120 $C exec -s "$SESSION" 2>/dev/null | grep -q alive; then
    date +%T >> "$DIR/keepalive.log"; sleep 180
  else
    echo "fail $(date +%T)" >> "$DIR/keepalive.log"; sleep 20
  fi
done
