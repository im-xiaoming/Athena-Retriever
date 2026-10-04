#!/bin/bash
# Watch the InternVideo2 extraction on Colab until it ends, secure the output, then stop the session.
#   setsid nohup tools/iv2_finish.sh [session] > /dev/null 2>&1 < /dev/null &
# Log: logs/iv2_finish.log. Final state (DONE / INCOMPLETE / NEEDS_ATTENTION): logs/iv2_finish.state
#
# 1. every 5 min: report progress; if the extractor died without EXTRACT_DONE, relaunch it
#    (it resumes, skipping finished videos), at most 3 times
# 2. on EXTRACT_DONE: check every .npz (broken ones are deleted and re-extracted), final rsync
#    to Drive, pack into tar shards and upload to HF (nguyenminh04/uniav-youcook2-data/iv2_feats)
# 3. once the data is safe on Drive or HF: stop the keepalive loop and the Colab session
# 4. download the shards from HF into data/youcookii/iv2_feats and verify the file count
# Needs on the VM: /content/iv2_finalize.py (tools/iv2_finalize_colab.py) and /content/.hf_token
SESSION=${1:-omni}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG=$ROOT/logs/iv2_finish.log
STATE=$ROOT/logs/iv2_finish.state
NAME=${IV2_NAME:-iv2_feats}   # IV2_NAME=iv2_feats_shift for the half-second shifted extraction
DEST=$ROOT/data/youcookii/$NAME
C=~/.local/bin/colab
HF=https://huggingface.co/datasets/nguyenminh04/uniav-youcook2-data/resolve/main/$NAME
exec 8>"$ROOT/logs/iv2_finish.lock"; flock -n 8 || { echo "already running"; exit 0; }

say() { echo "$(date '+%m-%d %H:%M:%S')  $*" >> "$LOG"; }
state() { echo "$1 $(date '+%m-%d %H:%M:%S')" > "$STATE"; say "STATE $1"; }

# run a shell command on the VM, print its stdout; empty output means the call failed
rexec() {
  printf 'import subprocess\nprint(subprocess.run(r"""%s""", shell=True, capture_output=True, text=True).stdout)\n' "$1" \
    | timeout 150 $C exec -s "$SESSION" 2>/dev/null
}

# start one finalize step in the background on the VM and wait for its _OK / _FAIL line
step() {
  local name=$1 limit=$2 out t=0
  rexec "nohup setsid env IV2_NAME=$NAME python3 /content/iv2_finalize.py $name > /content/out/iv2_fin_$name.log 2>&1 < /dev/null &" > /dev/null
  while [ $t -lt "$limit" ]; do
    sleep 60; t=$((t + 60))
    out=$(rexec "cat /content/out/iv2_fin_$name.log")
    if echo "$out" | grep -qE "${name^^}_(OK|FAIL)"; then
      echo "$out" | grep -v '^$' | sed 's/^/      /' >> "$LOG"
      echo "$out" | grep -q "${name^^}_OK"; return
    fi
  done
  say "$name: no result after $limit s"; return 1
}

relaunch() {
  say "relaunching the extractor (attempt $1/3)"
  rexec 'mv /content/out/iv2_full.log /content/out/iv2_full.log.$(date +%s); rm -f /content/out/iv2_full.done;
         cd /content && nohup setsid bash /content/iv2_full.sh > /dev/null 2>&1 < /dev/null &' > /dev/null
}

say "watcher started for session $SESSION"
state WATCHING
relaunches=0; complete=1
while true; do
  # ---- 1. wait for the extractor to finish -------------------------------------------------
  while true; do
    out=$(rexec 'if grep -q EXTRACT_DONE /content/out/iv2_full.log; then echo STATE_DONE;
                 elif pgrep -f "[i]v2_full.sh|[e]xtract_internvideo2" > /dev/null; then echo STATE_RUNNING;
                 else echo STATE_CRASHED; fi; grep -E "^[0-9:]+ [0-9]+/" /content/out/iv2_full.log | tail -1')
    st=$(echo "$out" | grep -o 'STATE_[A-Z]*' | head -1)
    case $st in
      STATE_RUNNING) say "running   $(echo "$out" | grep -E '^[0-9:]+ [0-9]+/' | cut -c1-120)" ;;
      STATE_DONE) say "extraction finished"; break ;;
      STATE_CRASHED)
        say "extractor is not running and the log has no EXTRACT_DONE:"
        rexec 'tail -5 /content/out/iv2_full.log' | grep -v '^$' | cut -c1-200 | sed 's/^/      /' >> "$LOG"
        if [ $relaunches -lt 3 ]; then
          relaunches=$((relaunches + 1)); relaunch $relaunches
        else
          say "gave up after 3 relaunches; securing what exists"; complete=0; break
        fi ;;
      *) say "no answer from the session (network?), retrying" ;;
    esac
    sleep 300
  done

  # ---- 2. check every file ---------------------------------------------------------------
  # the Drive backup loop of iv2_full.sh does one last rsync after the extractor exits
  for i in $(seq 15); do
    rexec 'cat /content/out/iv2_backup.log' | grep -q BACKUP_DONE && break; sleep 60
  done
  say "checking every .npz"
  if step check 1800; then break; fi
  bad=$(rexec 'python3 -c "import json; print(len(json.load(open(\"/content/out/iv2_check.json\"))[\"bad_removed\"]))"' | tr -d '[:space:]')
  if [ "${bad:-0}" != 0 ] && [ $relaunches -lt 3 ]; then
    say "$bad broken files removed, re-extracting them"
    relaunches=$((relaunches + 1)); relaunch $relaunches; sleep 300; continue
  fi
  say "check reports videos without features (see /content/out/iv2_check.json); securing what exists"
  complete=0; break
done

# ---- 3. secure the data, then stop the session ----------------------------------------------
say "final sync to Drive"
step sync 3600; drive_ok=$?
say "packing and uploading to HF"
step pack 3600; hf_ok=$?
if [ $hf_ok -ne 0 ]; then say "HF upload failed, one more try"; step pack 3600; hf_ok=$?; fi
say "drive: $([ $drive_ok -eq 0 ] && echo OK || echo FAIL)   hf: $([ $hf_ok -eq 0 ] && echo OK || echo FAIL)"

if [ $drive_ok -ne 0 ] && [ $hf_ok -ne 0 ]; then
  # last resort: pull the tar shards straight from the VM
  say "both copies failed; downloading the tar shards directly from the VM"
  mkdir -p "$DEST/_tar"; direct_ok=1
  for f in $(rexec 'ls /content/iv2_tar'); do
    timeout 1800 $C download -s "$SESSION" "/content/iv2_tar/$f" "$DEST/_tar/$f" > /dev/null 2>&1 || direct_ok=0
  done
  if [ $direct_ok -ne 1 ] || [ ! -s "$DEST/_tar/manifest.json" ]; then
    say "direct download failed too; the session is LEFT RUNNING so nothing is lost"
    state NEEDS_ATTENTION; exit 1
  fi
fi

say "stopping the keepalive loop and the Colab session $SESSION"
touch "$ROOT/logs/keepalive.stop"
rexec 'rm -f /content/.hf_token' > /dev/null
for i in 1 2 3; do
  timeout 120 $C stop -s "$SESSION" < /dev/null >> "$LOG" 2>&1 && break; sleep 30
done
timeout 60 $C sessions >> "$LOG" 2>&1

# ---- 4. local copy ------------------------------------------------------------------------
mkdir -p "$DEST/_tar"
if [ $hf_ok -eq 0 ]; then
  TOK=$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tr -d '\r\n ')
  say "downloading the shards from HF"
  curl -sfL --retry 5 -H "Authorization: Bearer $TOK" "$HF/manifest.json" -o "$DEST/_tar/manifest.json"
  for f in $(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['shards']))" "$DEST/_tar/manifest.json"); do
    curl -sfL --retry 5 -C - -H "Authorization: Bearer $TOK" "$HF/$f" -o "$DEST/_tar/$f" || say "download of $f failed"
  done
fi
for f in "$DEST"/_tar/*.tar; do tar -xf "$f" -C "$DEST" || say "cannot unpack $f"; done
want=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['n_files'])" "$DEST/_tar/manifest.json" 2>/dev/null)
have=$(ls "$DEST" | grep -c '\.npz$')
say "local copy: $have .npz files in $DEST (manifest: ${want:-?})"
if [ -n "$want" ] && [ "$have" -ge "$want" ]; then
  rm -rf "$DEST/_tar"
  [ $complete -eq 1 ] && state DONE || state INCOMPLETE
else
  say "local copy is short; the data is on Drive/HF, fetch it again"
  state NEEDS_ATTENTION
fi
