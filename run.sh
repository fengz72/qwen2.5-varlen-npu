#!/bin/bash

set -e

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

cd "$(dirname "$0")/atb/models/qwen2.5-0.5b"

./run.sh pass
./run.sh export --prune
./run.sh atc --dump
./run.sh infer --profiling
