# export DUMP_GRAPH_PATH="./atb/dump_graph"
# export DUMP_GE_GRAPH=2
# export DUMP_GRAPH_LEVEL=1

# export ASCEND_GLOBAL_LOG_LEVEL=0
# export ASCEND_SLOG_PRINT_TO_STDOUT=1

# python -m qwen_varlen.export_air --device 0 --dynamic --run-atc --soc Ascend910_9382 --debug

python -m qwen_varlen.export_air --device 0 --dynamic --run-atc --soc Ascend910_9382

./atb/build/bench_latency \
    --model atb/models/qwen2.5-0.5b/om/qwen2.5-0.5b_linux_aarch64.om \
    --threads 1 --requests 10 --warmup 10 --profiling --device-id 0