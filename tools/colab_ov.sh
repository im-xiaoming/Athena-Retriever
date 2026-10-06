#!/bin/bash
# Colab A100: the open-vocabulary runs of docs/PLAN_openvocab.md, one after the other.
#   bash tools/colab_ov.sh setup      # clone, install, build NMS, fetch YouCook2 + COIN features (HF token in /content/hf_token)
#   bash tools/colab_ov.sh queue R0 R1 R2 R3 R4
# Logs: /content/ov_logs/<run>.log (one table row per epoch). touch /content/ov_stop to stop after the current run;
# a run is stopped at once by killing its train_event.py.
set -e
REPO=/content/UniAV
case "$1" in
setup)
  cd /content
  [ -d UniAV ] || git clone -q -b ov-refine https://github.com/im-xiaoming/UniAV-fixed.git UniAV
  cd UniAV && git pull -q && git log --oneline -1
  pip install -q pycocoevalcap h5py tensorboard 2>&1 | tail -1
  java -version 2>&1 | head -1 || echo "no java: METEOR will be skipped"
  (cd libs/utils && python setup.py install --user > /content/nms_build.log 2>&1) && python -c "import nms_1d_cpu" && echo "nms ok"
  export HF_TOKEN=$(cat /content/hf_token)
  python - <<'PY'
import os, tarfile
from huggingface_hub import snapshot_download
d = snapshot_download('nguyenminh04/uniav-youcook2-data', repo_type='dataset', local_dir='/content/hf_yc',
                      allow_patterns=['iv2_feats/*', 'teacher/*'], token=os.environ['HF_TOKEN'])
os.makedirs('data/youcookii/iv2_feats', exist_ok=True)
for t in sorted(os.listdir(d + '/iv2_feats')):
    if t.endswith('.tar'):
        tarfile.open(d + '/iv2_feats/' + t).extractall('data/youcookii/iv2_feats')
os.system('cp %s/teacher/omni_emb_full.npz data/youcookii/' % d)
print('youcook2 features:', len(os.listdir('data/youcookii/iv2_feats')))
PY
  python tools/coin_data.py fetch --cache /content/hf_coin
  ;;
queue)
  shift
  cd $REPO
  mkdir -p /content/ov_logs
  rm -f /content/ov_stop
  for run in "$@"; do
    [ -f /content/ov_stop ] && { echo "stop file: queue ends before $run"; break; }
    case $run in
      R0) ARGS="" ;;                                                           # YouCook2 only, as configured
      R1) ARGS="coin.train=true" ;;
      R2) ARGS="coin.train=true model.text_proj=none" ;;
      R3) ARGS="coin.train=true model.text_proj=none model.train_cfg.loss_weight_other=1.0" ;;
      R4) ARGS="coin.train=true model.text_proj=none model.train_cfg.loss_weight_other=1.0 model.pyramid_attn=self" ;;
      *) echo "unknown run $run"; continue ;;
    esac
    echo "$(date +%H:%M:%S) start $run: $ARGS"
    python -u train_event.py configs/youcook2_event.yaml --output ov_$run --note "PLAN_openvocab $run" \
      --set num_workers=4 $ARGS > /content/ov_logs/$run.log 2>&1 || echo "$(date +%H:%M:%S) $run exited with $?"
    echo "$(date +%H:%M:%S) end $run"
  done
  ;;
*)
  echo "usage: $0 setup | queue R0 [R1 ...]"; exit 1 ;;
esac
