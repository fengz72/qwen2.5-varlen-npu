#!/bin/bash

set -e

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

cd "$(dirname "$0")/atb/models/qwen2.5-0.5b"

./run.sh pass
./run.sh export --prune
./run.sh atc
./run.sh bench --profiling --warmup 50 --requests 100
./run.sh parse
