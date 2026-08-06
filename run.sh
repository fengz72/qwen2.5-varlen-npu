#!/bin/bash

set -e

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=0

cd atb
bash build.sh

cd "$(dirname "$0")/models/qwen2.5-0.5b"

PYTHONPATH=../../..:.. python3 -m model.prepare_air_inputs --device 8 --prune-lm-head

./run.sh pass
./run.sh export --prune --prefix-len 25
./run.sh atc
./run.sh bench --profiling --warmup 5 --requests 10 --fixed-seq 150 --prefix-len 20 --device 4
./run.sh parse
