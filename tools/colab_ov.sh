#!/bin/bash
# Colab A100: the open-vocabulary runs of docs/PLAN_openvocab.md, one after the other.
#   BRANCH=multi-data bash tools/colab_ov.sh setup   # clone, install, build NMS, fetch YouCook2 + COIN + ActivityNet +
#                                                    # HT-Step features (HF token in /content/hf_token)
#   bash tools/colab_ov.sh queue R0 R1 R2 R3 R4
# Logs: /content/ov_logs/<run>.log (one table row per epoch). touch /content/ov_stop to stop after the current run;
# a run is stopped at once by killing its train.py. Two queues can run at once (the A100 is
# CPU-bound at ~27% GPU with one run): bash tools/colab_ov.sh queue E1 E2
set -e
REPO=/content/Athena-Retriever
case "$1" in
setup)
  cd /content
  [ -d Athena-Retriever ] || git clone -q -b ${BRANCH:-multi-data} https://github.com/im-xiaoming/Athena-Retriever.git Athena-Retriever
  cd Athena-Retriever && git pull -q && git log --oneline -1
  pip install -q pycocoevalcap h5py tensorboard 2>&1 | tail -1
  java -version 2>&1 | head -1 || echo "no java: METEOR will be skipped"
  (cd libs/utils && python setup.py install --user > /content/nms_build.log 2>&1) && python -c "import torch, nms_1d_cpu" && echo "nms ok"
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
  if [ -n "$IV2_MID" ]; then   # InternVideo2 middle-layer features (m20/m30/m35) and their training-set mean
    python - <<'PY'
import os
from huggingface_hub import snapshot_download, HfApi
d = snapshot_download('nguyenminh04/uniav-youcook2-data', repo_type='dataset', local_dir='/content/hf_yc',
                      allow_patterns=['iv2_mid/*', 'iv2_mid_mean.npz'], token=os.environ['HF_TOKEN'])
os.system('ln -sfn %s/iv2_mid data/youcookii/iv2_mid' % d)
print('youcook2 iv2_mid:', len(os.listdir('data/youcookii/iv2_mid')))
if os.path.exists(d + '/iv2_mid_mean.npz'):
    os.system('cp %s/iv2_mid_mean.npz data/youcookii/' % d)
else:
    os.system('python tools/iv2_mid_mean.py --upload')
PY
  fi
  [ -n "$YC_ONLY" ] && exit 0   # YouCook2-only runs: no COIN / ActivityNet / HT-Step (their features lack m* keys)
  python tools/coin_data.py fetch --cache /content/hf_coin
  python tools/extra_data.py fetch anet --cache /content/hf_anet
  python tools/extra_data.py fetch htstep --cache /content/hf_htstep
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
      ALL) ARGS="coin.train=true coin.batch_ratio=0.2 extra_train.anet.ratio=0.3 extra_train.htstep.ratio=0.1" ;;  # every training set
      E1) ARGS="train_cfg.ema_decay=0.999" ;;                               # localization options on R0
      E2) ARGS="dataset.center_sample_radius=1.5" ;;
      # InternVideo2 middle layers (setup with IV2_MID=1 YC_ONLY=1); *_s2 = second seed
      M35) ARGS="train_cfg.ema_decay=0.999 dataset.iv2_video_keys=[m35]" ;;
      M35_s2) ARGS="train_cfg.ema_decay=0.999 dataset.iv2_video_keys=[m35] init_rand_seed=1234567892" ;;
      VM35) ARGS="train_cfg.ema_decay=0.999 dataset.iv2_video_keys=[v768,m35]" ;;
      VM35_s2) ARGS="train_cfg.ema_decay=0.999 dataset.iv2_video_keys=[v768,m35] init_rand_seed=1234567892" ;;
      M30) ARGS="train_cfg.ema_decay=0.999 dataset.iv2_video_keys=[m30]" ;;
      M20) ARGS="train_cfg.ema_decay=0.999 dataset.iv2_video_keys=[m20]" ;;
      E1_s2) ARGS="train_cfg.ema_decay=0.999 init_rand_seed=1234567892" ;;
      *) echo "unknown run $run"; continue ;;
    esac
    echo "$(date +%H:%M:%S) start $run: $ARGS"
    python -u train.py configs/youcook2_event.yaml --output ov_$run --note "PLAN_openvocab $run" \
      --epochs ${EPOCHS:-10} ${PATIENCE:+--patience $PATIENCE} --set num_workers=4 $ARGS > /content/ov_logs/$run.log 2>&1 || echo "$(date +%H:%M:%S) $run exited with $?"
    echo "$(date +%H:%M:%S) end $run"
  done
  ;;
*)
  echo "usage: $0 setup | queue R0 [R1 ...]"; exit 1 ;;
esac
