# export DUMP_GRAPH_PATH="./atb/dump_graph"
# export DUMP_GE_GRAPH=2
# export DUMP_GRAPH_LEVEL=1

# export ASCEND_GLOBAL_LOG_LEVEL=0
# export ASCEND_SLOG_PRINT_TO_STDOUT=1

# python -m qwen_varlen.export_air --device 0 --dynamic --run-atc --soc Ascend910_9382 --debug

# 1. 导出模型
python -m qwen_varlen.export_air --device 0 --dynamic --run-atc --soc Ascend910_9382

# 2. 生成golden
python -m qwen_varlen.prepare_air_inputs --device 0

# 3. 精度测试
cd atb
./build/acl_infer \
    --model models/qwen2.5-0.5b/om/qwen2.5-0.5b_linux_aarch64.om \
    --output_dir models/qwen2.5-0.5b/output \
    --device_id 0 \
    --input "arg1_1:10:int64:ND:models/qwen2.5-0.5b/input_data/actual_seq_lengths.bin" \
    --input "arg3_1:1,2080,64:float16:ND:models/qwen2.5-0.5b/input_data/cos.bin" \
    --input "arg5_1:1,2080,64:float16:ND:models/qwen2.5-0.5b/input_data/sin.bin" \
    --input "arg8_1:2080:int64:ND:models/qwen2.5-0.5b/input_data/input_ids.bin"

# 4. 对比精度
python3 tools/compare.py \
    models/qwen2.5-0.5b/input_data/golden_logits.bin \
    models/qwen2.5-0.5b/output/output_0.bin \
    --dtype float16 \
    --rtol 1e-2 --atol 1e-2

# 5. 性能测试
./build/bench_latency \
    --model models/qwen2.5-0.5b/om/qwen2.5-0.5b_linux_aarch64.om \
    --sweep 1,2,3,4,5,6,7,8 --requests 8000 --warmup 50  --device-id 0
