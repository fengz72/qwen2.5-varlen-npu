./run.sh -m qwen2.5-0.5b --skip-atc --profiling --warmup 10 --bench 100

python3 tools/parse_profiling.py parse-and-export --profiling_dir atb/models/qwen2.5-0.5b/profiling_data