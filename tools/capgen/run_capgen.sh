#!/bin/bash
# Train and score the caption generators one after another (logs/capgen_<name>.log).
#   setsid nohup tools/capgen/run_capgen.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/../.." || exit 1
PY=/home/minh/uniav-api-env/bin/python
$PY tools/capgen/train_capgen.py --name prefix > logs/capgen_prefix.log 2>&1
$PY tools/capgen/train_capgen.py --name rag5 --rag 5 > logs/capgen_rag5.log 2>&1
echo CAPGEN_DONE >> logs/capgen_rag5.log
