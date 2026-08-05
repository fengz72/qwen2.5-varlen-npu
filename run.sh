#!/bin/bash

set -e

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

cd atb
bash build.sh

cd "$(dirname "$0")/models/qwen2.5-0.5b"

./run.sh pass
./run.sh export --prune
./run.sh atc
./run.sh bench --profiling --warmup 50 --requests 100 --fixed-seq 150
./run.sh bench --profiling --warmup 50 --requests 100 --fixed-seq 218
./run.sh parse
