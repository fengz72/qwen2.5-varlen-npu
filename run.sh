#!/bin/bash

set -e

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

cd atb
bash build.sh

cd "$(dirname "$0")/models/qwen2.5-0.5b"

./run.sh pass
./run.sh export --prune --prefix-len 25
./run.sh atc
./run.sh bench --profiling --warmup 50 --requests 100 --fixed-seq 150 --prefix-len 20 --device 4
./run.sh bench --profiling --warmup 50 --requests 100 --fixed-seq 218 --prefix-len 25 --device 4
./run.sh parse
