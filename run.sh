#!/bin/bash

set -e

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译推理脚本
cd atb
bash build.sh

cd "$(dirname "$0")/models/qwen2.5-0.5b"

# 生成输入
PYTHONPATH=../../..:.. python3 -m model.prepare_air_inputs --device 8 --prune-lm-head


./run.sh pass # 编译安装pass
./run.sh export --prune # 导出air
./run.sh atc # 导出om
./run.sh bench --profiling --warmup 50 --requests 100
./run.sh parse
