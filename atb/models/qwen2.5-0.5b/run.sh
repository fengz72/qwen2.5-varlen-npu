#!/bin/bash
# =============================================================================
# run.sh — qwen2.5-0.5b 全流程一键脚本
#
# 子命令:
#   pass      编译 NZ weight pass 并安装到 CANN vendor 目录
#   export    导出 AIR (PyTorch → AIR)
#   atc       ATC 编译 (AIR → OM, 调用 export_air.py --skip-export --run-atc)
#   infer     推理验证 (acl_infer)
#   bench     延迟 benchmark (bench_latency)
#   parse     解析 profiling 数据
#   all       全流程: pass → export → atc → infer
#
# 通用选项:
#   --device N        NPU 设备号 (默认 8)
#   --soc XXX         SoC 型号 (默认 Ascend910_9382)
#   --warmup N        预热次数 (默认 10)
#   --requests N      bench 请求数 (默认 100)
#   --threads N       bench 线程数 (默认 1)
#   --prune           开启 lm_head vocab 剪裁
#   --profiling       开启 profiling (infer/bench 生效)
#   --debug           ATC 编译开启 --log=debug
#   --dump            ATC 编译开启 GE 图 dump
#   --aicore-num N    限制运行时 AICore 数量 (如 12)
#   --dry-run         只打印命令不执行
#   -h, --help        显示帮助
#
# 示例:
#   ./run.sh all --profiling
#   ./run.sh pass && ./run.sh atc && ./run.sh bench --profiling
#   ./run.sh bench --warmup 10 --requests 100 && ./run.sh parse
#   ./run.sh export --prune
# =============================================================================

set -e

# ---- 路径常量 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}"
ATB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${ATB_DIR}/.." && pwd)"
MODEL_NAME="qwen2.5-0.5b"
AIR_DIR="${MODEL_DIR}/air"
OM_DIR="${MODEL_DIR}/om"
PASS_DIR="${MODEL_DIR}/pass"
PROFILING_DIR="${MODEL_DIR}/profiling_data"
AIR_PATH="${AIR_DIR}/${MODEL_NAME}.air"

# ---- 默认参数 ----
DEVICE=8
SOC="Ascend910_9382"
WARMUP=10
REQUESTS=100
THREADS=1
PRUNE=false
PROFILING=false
DEBUG=false
DUMP=false
AICORE_NUM=""
DRY_RUN=false

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "\n${BLUE}==== $* ====${NC}"; }

usage() {
    cat << 'EOF'
Usage: ./run.sh <subcommand> [options]

Subcommands:
  pass      Build NZ weight pass and install to CANN vendor dir
  export    Export AIR (PyTorch → AIR)
  atc       ATC compile (AIR → OM, via export_air.py --skip-export --run-atc)
  infer     Inference verification (acl_infer)
  bench     Latency benchmark (bench_latency)
  parse     Parse profiling data
  all       Full pipeline: pass → export → atc → infer

Options:
  --device N        NPU device ID (default: 8)
  --soc XXX         SoC version (default: Ascend910_9382)
  --warmup N        Warmup runs (default: 10)
  --requests N      Benchmark requests (default: 100)
  --threads N       Benchmark threads (default: 1)
  --prune           Enable lm_head vocab pruning
  --profiling       Enable profiling
  --debug           ATC --log=debug
  --dump            ATC GE graph dump
  --aicore-num N    Limit AICore count (e.g. 12)
  --dry-run         Print commands without executing
  -h, --help        Show this help

Examples:
  ./run.sh all --profiling
  ./run.sh pass && ./run.sh atc && ./run.sh bench --profiling
  ./run.sh bench --warmup 10 --requests 100 && ./run.sh parse
  ./run.sh export --prune
EOF
}

# =============================================================================
# 通用工具
# =============================================================================
parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device)     DEVICE="$2"; shift 2 ;;
            --soc)        SOC="$2"; shift 2 ;;
            --warmup)     WARMUP="$2"; shift 2 ;;
            --requests)   REQUESTS="$2"; shift 2 ;;
            --threads)    THREADS="$2"; shift 2 ;;
            --prune)      PRUNE=true; shift ;;
            --profiling)  PROFILING=true; shift ;;
            --debug)      DEBUG=true; shift ;;
            --dump)       DUMP=true; shift ;;
            --aicore-num) AICORE_NUM="$2"; shift 2 ;;
            --dry-run)    DRY_RUN=true; shift ;;
            -h|--help)    usage; exit 0 ;;
            *)            log_error "Unknown option: $1"; usage; exit 1 ;;
        esac
    done
}

run_or_echo() {
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] $*"
    else
        eval "$@"
    fi
}

prof_flag() {
    [ "$PROFILING" = true ] && echo "--profiling" || echo ""
}

# =============================================================================
# 1. 编译 NZ weight pass
# =============================================================================
do_pass() {
    log_step "Step 1: Build NZ weight pass"
    cd "${PASS_DIR}"
    run_or_echo "bash build.sh"
    log_info "NZ weight pass built and installed"
}

# =============================================================================
# 2. 导出 AIR
# =============================================================================
do_export() {
    log_step "Step 2: Export AIR (PyTorch → AIR)"
    local prune_flag=""
    [ "$PRUNE" = true ] && prune_flag="--prune-lm-head"
    cd "${MODEL_DIR}"
    run_or_echo "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH} python3 -m model.export_air \
        --device ${DEVICE} \
        --output-dir ${AIR_DIR} \
        --om-dir ${OM_DIR} \
        --model-name ${MODEL_NAME} \
        --soc ${SOC} ${prune_flag}"
    if [ "$DRY_RUN" != true ]; then
        [ -f "${AIR_PATH}" ] && log_info "AIR exported: ${AIR_PATH}" || log_error "AIR not found: ${AIR_PATH}"
    fi
}

# =============================================================================
# 3. ATC 编译 (AIR → OM)
# =============================================================================
do_atc() {
    log_step "Step 3: ATC compile (AIR → OM)"
    if [ ! -f "${AIR_PATH}" ]; then
        log_error "AIR not found: ${AIR_PATH}, run './run.sh export' first"
        exit 1
    fi
    local debug_flag=""
    [ "$DEBUG" = true ] && debug_flag="--debug"
    local aicore_flag=""
    [ -n "$AICORE_NUM" ] && aicore_flag="--aicore-num ${AICORE_NUM}"
    local dump_env=""
    if [ "$DUMP" = true ]; then
        local dump_dir="${MODEL_DIR}/dump_graph"
        mkdir -p "$dump_dir"
        dump_env="DUMP_GRAPH_PATH=${dump_dir} DUMP_GE_GRAPH=2 DUMP_GRAPH_LEVEL=2 PRINT_MODEL=1"
        log_info "GE graph dump enabled → ${dump_dir}"
    fi
    cd "${MODEL_DIR}"
    run_or_echo "${dump_env} PYTHONPATH=${REPO_ROOT}:${PYTHONPATH} python3 -m model.export_air \
        --skip-export \
        --run-atc \
        --output-dir ${AIR_DIR} \
        --om-dir ${OM_DIR} \
        --model-name ${MODEL_NAME} \
        --soc ${SOC} ${debug_flag} ${aicore_flag}"
}

# =============================================================================
# 4. 推理验证 (调用 atb/run.sh)
# =============================================================================
do_infer() {
    log_step "Step 4: Inference verification (acl_infer)"
    cd "${ATB_DIR}"
    run_or_echo "./run.sh -m ${MODEL_NAME} \
        --skip-build \
        --skip-atc \
        --warmup ${WARMUP} \
        --device-id ${DEVICE} $(prof_flag)"
}

# =============================================================================
# 5. 延迟 benchmark
# =============================================================================
do_bench() {
    log_step "Step 5: Latency benchmark (bench_latency)"

    # 查找 OM 文件 (ATC 可能添加系统后缀)
    local om_file="${OM_DIR}/${MODEL_NAME}_linux_aarch64.om"
    if [ ! -f "$om_file" ]; then
        om_file=$(find "${OM_DIR}" -name "${MODEL_NAME}*.om" -type f 2>/dev/null | head -1)
    fi
    if [ ! -f "$om_file" ]; then
        log_error "OM not found in ${OM_DIR}, run './run.sh atc' first"
        exit 1
    fi

    cd "${ATB_DIR}"
    run_or_echo "./build/bench_latency \
        --model ${om_file} \
        --threads ${THREADS} \
        --requests ${REQUESTS} \
        --warmup ${WARMUP} \
        --device-id ${DEVICE} $(prof_flag)"
}

# =============================================================================
# 6. 解析 profiling 数据
# =============================================================================
do_parse() {
    log_step "Step 6: Parse profiling data"
    if [ ! -d "${PROFILING_DIR}" ]; then
        log_error "Profiling dir not found: ${PROFILING_DIR}"
        log_error "Run with --profiling first"
        exit 1
    fi
    cd "${ATB_DIR}"
    run_or_echo "python3 tools/parse_profiling.py parse-and-export \
        --profiling_dir ${PROFILING_DIR}"
}

# =============================================================================
# 全流程
# =============================================================================
do_all() {
    do_pass
    do_export
    do_atc
    do_infer
    log_info "All steps completed successfully"
}

# =============================================================================
# 主入口
# =============================================================================
if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

SUBCMD="$1"
shift

case "$SUBCMD" in
    pass)    parse_common_args "$@"; do_pass ;;
    export)  parse_common_args "$@"; do_export ;;
    atc)     parse_common_args "$@"; do_atc ;;
    infer)   parse_common_args "$@"; do_infer ;;
    bench)   parse_common_args "$@"; do_bench ;;
    parse)   parse_common_args "$@"; do_parse ;;
    all)     parse_common_args "$@"; do_all ;;
    -h|--help) usage; exit 0 ;;
    *)       log_error "Unknown subcommand: $SUBCMD"; usage; exit 1 ;;
esac
