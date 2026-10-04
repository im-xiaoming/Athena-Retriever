#!/bin/bash
# Variants of the InternVideo2 model (run iv2), trained one after another.
#   setsid nohup tools/run_queue_variants.sh > /dev/null 2>&1 < /dev/null &
# Progress: logs/variants_queue.log, one log per run in logs/<run>.log
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1
PY=/home/minh/uniav-env/bin/python
Q=logs/variants_queue.log
TEACHER=dataset.omni_emb_file=./data/youcookii/omni_emb_full.npz
IV2=dataset.feat_source=iv2
exec 7>logs/variants_queue.lock; flock -n 7 || { echo "already running"; exit 0; }
say() { echo "$(date '+%m-%d %H:%M:%S')  $*" >> "$Q"; }
final() { grep '^Final eval' "logs/$1.log" | grep -oE '(R@0.5|R@0.7|ret_sim|ret_sim\[top1\]|CIDEr|METEOR) [0-9.]+' | tr '\n' ' '; }

run() {   # run <name> <note> <train_event.py arguments...>
  local name=$1 note=$2; shift 2
  if grep -qs '"final_eval": {' "experiments/runs/$(hostname)-$name.json"; then
    say "$name already finished, skipped"; return
  fi
  while pgrep -f "[t]rain_event.py" > /dev/null; do sleep 60; done   # one training at a time (RAM)
  say "start $name: $*"
  $PY train_event.py configs/youcook2_event.yaml --output "$name" --note "$note" "$@" > "logs/$name.log" 2>&1
  say "end   $name (exit $?): $(final "$name")"
}

# 0. inference settings on the API checkpoint: no training
if [ ! -s logs/iv2_infer_sweep.log ]; then
  say "inference sweep on ckpt/iv2/best_cap (iou_power x soft-NMS sigma)"
  $PY train_event.py configs/youcook2_event.yaml --output iv2 --eval ckpt/iv2/best_cap.pth.tar --fast \
    --set "$TEACHER" "$IV2" --iou-power 0.2,0.3,0.5 --nms soft:0.7:0.5,soft:0.7:0.9 > logs/iv2_infer_sweep.log 2>&1
  grep '^iou_power' logs/iv2_infer_sweep.log | while read -r l; do
    say "  $(echo "$l" | grep -oE '^iou_power [0-9.]+ +NMS [^ ]+|(R@0.5|R@0.7|ret_sim|CIDEr) [0-9.]+' | tr '\n' ' ')"
  done
fi

run iv2_noteach 'iv2 without the OmniRetriever teacher' --set "$IV2"
run iv2_reg 'iv2, dropout 0.1 and drop-path 0.2 (segmentation overfits after epoch 7)' \
  --set "$TEACHER" "$IV2" model.train_cfg.dropout=0.1 model.train_cfg.droppath=0.2
run iv2_emb04 'iv2, embedding and span loss weights 0.4 (ret_sim still rising at the last epoch)' \
  --set "$TEACHER" "$IV2" model.train_cfg.loss_weight_emb=0.4 model.train_cfg.loss_weight_span=0.4
run iv2_ep8 'iv2, 8 epochs instead of 10 (cosine ends before the overfitting)' \
  --epochs 8 --set "$TEACHER" "$IV2"
# round 2: confirm the 8-epoch schedule, and combine it with the regularisation
run iv2_ep8_seed2 'iv2_ep8, second seed' --epochs 8 --set "$TEACHER" "$IV2" init_rand_seed=2024
run iv2_ep8_reg 'iv2, 8 epochs + dropout 0.1, drop-path 0.2' \
  --epochs 8 --set "$TEACHER" "$IV2" model.train_cfg.dropout=0.1 model.train_cfg.droppath=0.2

# round 3: caption space without ONE-PEACE, teacher and loss changes from the OmniRetriever paper, capacity
run txt_omni 'iv2, captions trained and picked in the OmniRetriever text space instead of ONE-PEACE' \
  --set "$TEACHER" "$IV2" dataset.caption_space=omni
run teach_tva 'iv2, teacher target = clip audio+video fused with its caption (T+V+A)' \
  --set "$TEACHER" "$IV2" model.train_cfg.omni_target=tva
run modal_aux 'iv2, fusion-as-teacher inside the model: video-only and audio-only segment vectors (weight 0.1)' \
  --set "$TEACHER" "$IV2" model.train_cfg.loss_weight_modal=0.1
run wide 'iv2, embedding and head width 768 instead of 512 (299M parameters)' \
  --set "$TEACHER" "$IV2" model.embd_dim=768 model.head_dim=768

$PY tools/summarize_runs.py >> "$Q" 2>&1
say "QUEUE_DONE"
