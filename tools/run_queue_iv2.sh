#!/bin/bash
# Train the InternVideo2 comparison runs one after another once the features are local.
#   setsid nohup tools/run_queue_iv2.sh > /dev/null 2>&1 < /dev/null &
# Waits for tools/iv2_finish.sh to report DONE or INCOMPLETE, then runs the queue below with
# the full OmniRetriever teacher (same as omni100 / omni100_seed2), a second seed of the run
# with the best ret_sim, and refreshes experiments/RESULTS.md.
# Progress: logs/iv2_queue.log, one log per run in logs/<run>.log
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1
PY=/home/minh/uniav-env/bin/python
Q=logs/iv2_queue.log
TEACHER=dataset.omni_emb_file=./data/youcookii/omni_emb_full.npz
exec 7>logs/iv2_queue.lock; flock -n 7 || { echo "already running"; exit 0; }
say() { echo "$(date '+%m-%d %H:%M:%S')  $*" >> "$Q"; }

say "waiting for the InternVideo2 features (logs/iv2_finish.state)"
while true; do
  st=$(cut -d' ' -f1 logs/iv2_finish.state 2>/dev/null)
  case $st in
    DONE|INCOMPLETE) break ;;
    NEEDS_ATTENTION) say "feature copy needs attention, queue not started"; exit 1 ;;
  esac
  sleep 120
done
say "features ready: $(ls data/youcookii/iv2_feats | grep -c '\.npz$') files ($st)"

run() {   # run <name> <note> <overrides...>
  local name=$1 note=$2; shift 2
  if [ -f "experiments/runs/$(hostname)-$name.json" ] && grep -q '"final_eval": {' "experiments/runs/$(hostname)-$name.json"; then
    say "$name already finished, skipped"; return
  fi
  say "start $name: $*"
  $PY train_event.py configs/youcook2_event.yaml --output "$name" --set "$TEACHER" "$@" --note "$note" \
    > "logs/$name.log" 2>&1
  say "end   $name (exit $?): $(grep -E '^Best' "logs/$name.log" | tr -s ' ' | tr '\n' ' ')"
}

run iv2 'InternVideo2 v768 + BEATs a768 only, L2-normalised' dataset.feat_source=iv2
run iv2op 'InternVideo2 + ONE-PEACE, channels concatenated' dataset.feat_source=iv2+onepeace
run iv2_v512 'InternVideo2 v768 + text-aligned v512 + a768' dataset.feat_source=iv2 'dataset.iv2_video_keys=[v768,v512]'

best=$($PY - <<'EOF'
import json, socket
runs = {'iv2': ['dataset.feat_source=iv2'],
        'iv2op': ['dataset.feat_source=iv2+onepeace'],
        'iv2_v512': ['dataset.feat_source=iv2', 'dataset.iv2_video_keys=[v768,v512]']}
score = {}
for r in runs:
    try:
        score[r] = json.load(open('experiments/runs/%s-%s.json' % (socket.gethostname(), r)))['final_eval']['ret_sim']
    except Exception:
        pass
print(max(score, key=score.get) if score else '')
EOF
)
case $best in
  iv2) run iv2_seed2 'iv2, second seed' dataset.feat_source=iv2 init_rand_seed=2024 ;;
  iv2op) run iv2op_seed2 'iv2op, second seed' dataset.feat_source=iv2+onepeace init_rand_seed=2024 ;;
  iv2_v512) run iv2_v512_seed2 'iv2_v512, second seed' dataset.feat_source=iv2 'dataset.iv2_video_keys=[v768,v512]' init_rand_seed=2024 ;;
  *) say "no finished run to repeat" ;;
esac

$PY tools/summarize_runs.py >> "$Q" 2>&1
say "QUEUE_DONE"
