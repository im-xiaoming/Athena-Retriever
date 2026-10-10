#!/bin/bash
# Every 5 min: check that the Colab job (extraction) is alive; else relaunch on the first account that grants a T4.
# Free accounts first, the Pro account (default HOME) last. Stop: touch logs/watchdog.stop
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
C=/home/minh/.local/bin/colab
mkdir -p logs; rm -f logs/watchdog.stop
CUR=/home/minh   # account that holds the session
alive() { HOME=$CUR timeout 120 $C exec -s t4 <<<'import subprocess;print("OK" if subprocess.getoutput("pgrep -f colab_dl_extract.sh|head -1") else "NO")' 2>/dev/null | grep -q OK; }
while [ ! -f logs/watchdog.stop ]; do
  if ! alive; then
    echo "$(date +%T) dead on $CUR, relaunching" >> logs/watchdog.log
    touch logs/keepalive.stop; sleep 5
    for h in /home/minh/colab3 /home/minh/colab2 /home/minh; do
      HOME=$h $C stop -s t4 >/dev/null 2>&1
      if HOME=$h bash tools/colab_dl_launch.sh >> logs/watchdog.log 2>&1; then CUR=$h; echo "$(date +%T) running on $h" >> logs/watchdog.log; break; fi
    done
  fi
  sleep 300
done
