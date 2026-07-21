#!/bin/bash
# =============================================================================
# run.sh - One-click script for ATC model conversion and offline inference
#
# Usage:
#   ./run.sh -m <model_name> [options]
#
# Options:
#   -m, --model <name>    Model name (required, must exist in models/<name>/)
#   --skip-build          Skip building the inference binary
#   --skip-atc            Skip ATC model conversion
#   --atc-only            Only run ATC conversion (skip inference)
#   --infer-only          Only run inference (skip build and ATC)
#   --dump                Enable data dump during inference
#   --dump-stats          Enable stats dump (shortcut for dump_data=stats)
#   --async               Use async inference mode
#   --device-id <id>      NPU device ID (default: 0)
#   --list                List available models
#   --dry-run             Print commands without executing
#   -h, --help            Show this help message
#
# Examples:
#   ./run.sh -m model                    # Full pipeline: build + ATC + infer
#   ./run.sh -m model --skip-atc         # Build + infer (skip ATC)
#   ./run.sh -m model --dump             # Full pipeline with dump enabled
#   ./run.sh -m model --infer-only       # Only run inference
#   ./run.sh --list                      # List available models
# =============================================================================

set -e

export DUMP_GRAPH_PATH="./dump_graph"
export PRINT_MODEL=1
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=2

plogFlag=false
if [ $plogFlag == true ]; then
  echo "plog open!!!"
  export ASCEND_GLOBAL_EVENT_ENABLE=1
  export ASCEND_GLOBAL_LOG_LEVEL=0
  export ASCEND_SLOG_PRINT_TO_STDOUT=1
  export ASCEND_PROCESS_LOG_PATH=./plog
  export ASCEND_HOST_FILE_NUM=1000
fi

# ---- Color output ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "\n${BLUE}==== $* ====${NC}"; }

# ---- Resolve paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
BUILD_DIR="${SCRIPT_DIR}/build"
INFER_BIN="${BUILD_DIR}/acl_infer"

# ---- Default CLI options ----
MODEL_NAME=""
SKIP_BUILD=true
SKIP_ATC=false
ATC_ONLY=false
INFER_ONLY=false
DUMP_ENABLED=false
DUMP_STATS=false
DUMP_MODE=""
DUMP_LEVEL=""
DUMP_LAYERS=""
DUMP_PATH_OVERRIDE=""
PROFILING_ENABLED=false
USE_ASYNC=false
DEVICE_ID=0
DRY_RUN=false
LIST_MODELS=false
WARMUP_RUNS=0
BENCH_RUNS=1

# =============================================================================
# Functions
# =============================================================================

usage() {
    cat << 'EOF'
Usage: ./run.sh -m <model_name> [options]

One-click script for ATC model conversion and offline inference.

Options:
  -m, --model <name>    Model name (required, directory: models/<name>/)
  --skip-build          Skip building the inference binary
  --skip-atc            Skip ATC model conversion
  --atc-only            Only run ATC conversion
  --infer-only          Only run inference (skip build and ATC)
  --dump                Enable data dump during inference
  --dump-stats          Dump statistics instead of tensor data
  --dump-mode <mode>    Dump mode: input|output|all (default: from config or output)
  --dump-level <level>  Dump level: op|kernel|all (default: from config or op)
  --dump-layers <list>  Comma-separated layer names to dump (overrides config)
  --dump-path <dir>     Dump output directory (overrides default model dir)
  --profiling           Enable performance profiling during inference
  --async               Use async inference mode
  --device-id <id>      NPU device ID (default: 0)
  --warmup <N>          Number of warmup runs before timing (default: 0)
  --bench <N>           Number of benchmark runs for timing statistics (default: 1)
  --list                List available models
  --dry-run             Print commands without executing
  -h, --help            Show this help message

Model Directory Structure:
  models/<name>/
  ├── config            Model configuration file (required)
  ├── onnx/             Source model files (.onnx, .pb, etc.)
  ├── input_data/       Input binary files (.bin)
  ├── om/               Converted .om model (auto-generated)
  ├── output/           Inference output (auto-generated)
  ├── dump_data/        Dump data (auto-generated when --dump)
  └── profiling_data/   Profiling data (auto-generated when --profiling)

Examples:
  ./run.sh -m bert                     # Full pipeline
  ./run.sh -m bert --skip-atc          # Skip ATC, run inference
  ./run.sh -m bert --dump              # With data dump (output mode)
  ./run.sh -m bert --dump --dump-mode all --dump-level op  # Dump all I/O at op level
  ./run.sh -m bert --dump --dump-layers "Softmax,MatMul"   # Dump specific layers
  ./run.sh -m bert --dump --dump-path /tmp/dump            # Custom dump directory
  ./run.sh -m bert --dump-stats        # Dump statistics only
  ./run.sh -m bert --profiling         # With performance profiling
  ./run.sh -m bert --warmup 10 --bench 50  # Benchmark: 10 warmup + 50 timed runs
  ./run.sh -m resnet50 --atc-only      # Only convert model
  ./run.sh --list                      # List available models

Profiling Data Analysis:
  # List profiling sessions
  python3 tools/parse_profiling.py list --profiling_dir ./profiling_data

  # Show profiling summary
  python3 tools/parse_profiling.py summary --profiling_dir ./profiling_data

  # Show specific operator details
  python3 tools/parse_profiling.py show --profiling_dir ./profiling_data --op "Softmax"

  # Export to CSV
  python3 tools/parse_profiling.py export --profiling_dir ./profiling_data --output profiling.csv

  # Plot execution time
  python3 tools/parse_profiling.py plot --profiling_dir ./profiling_data --output profiling.png
EOF
}

list_models() {
    echo "Available models in ${MODELS_DIR}/:"
    echo ""
    if [ ! -d "${MODELS_DIR}" ]; then
        echo "  (no models directory)"
        return
    fi
    local found=false
    for dir in "${MODELS_DIR}"/*/; do
        [ -d "$dir" ] || continue
        local name=$(basename "$dir")
        [ "$name" = ".template" ] && continue
        if [ -f "${dir}/config" ]; then
            local model_name=$(grep "^MODEL_NAME=" "${dir}/config" | cut -d= -f2-)
            local model_file=$(grep "^MODEL_FILE=" "${dir}/config" | cut -d= -f2-)
            local input_count=$(grep "^INPUT_COUNT=" "${dir}/config" | cut -d= -f2-)
            local model_type=$(grep "^MODEL_TYPE=" "${dir}/config" | cut -d= -f2-)
            model_type="${model_type:-dynamic}"
            printf "  %-20s name=%-15s type=%-8s inputs=%-3s model=%s\n" \
                "$name" "${model_name:-?}" "$model_type" "${input_count:-?}" "${model_file:-?}"
            found=true
        fi
    done
    if [ "$found" = false ]; then
        echo "  (no models found)"
        echo ""
        echo "  To create a new model:"
        echo "    mkdir -p models/<your_model>"
        echo "    cp models/.template/config models/<your_model>/config"
        echo "    # Edit the config file with your model settings"
    fi
}

# Parse config file into shell variables
# Supports variable expansion: ${VAR_NAME} references are expanded
# Usage: load_config <config_file>
load_config() {
    local config_file="$1"
    
    # First pass: export all variables (including dimension variables)
    while IFS='=' read -r key value; do
        key=$(echo "$key" | xargs)
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        export "${key}=${value}"
    done < "$config_file"
    
    # Second pass: export with CFG_ prefix and expand variable references
    while IFS='=' read -r key value; do
        key=$(echo "$key" | xargs)
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        # Expand ${VAR} references using eval
        expanded_value=$(eval echo "\"${value}\"")
        export "CFG_${key}=${expanded_value}"
    done < "$config_file"
}

# Get config value with optional default
cfg() {
    local key="CFG_$1"
    local default="${2:-}"
    local val="${!key:-$default}"
    echo "$val"
}

# Resolve path: if relative, prepend model dir
resolve_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "${MODEL_DIR}/${path}"
    fi
}

# =============================================================================
# Step 1: Build inference binary
# =============================================================================
do_build() {
    log_step "Step 1: Build inference binary"

    if [ -f "${INFER_BIN}" ] && [ "$SKIP_BUILD" = true ]; then
        log_info "Binary exists and --skip-build specified, skipping"
        return 0
    fi

    if [ -f "${INFER_BIN}" ]; then
        log_info "Binary already exists: ${INFER_BIN}"
        read -p "Rebuild? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping build"
            return 0
        fi
    fi

    log_info "Building..."
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] ${SCRIPT_DIR}/build.sh"
    else
        "${SCRIPT_DIR}/build.sh"
    fi
    log_info "Build complete: ${INFER_BIN}"
}

# =============================================================================
# Step 2: ATC model conversion
# =============================================================================
do_atc() {
    log_step "Step 2: ATC model conversion"

    local model_file=$(cfg MODEL_FILE)
    local model_path=$(resolve_path "$model_file")

    if [[ "$model_file" == *.om ]]; then
        log_info "Model file is already .om, skipping ATC conversion"
        CFG_OM_PATH="$model_path"
        export CFG_OM_PATH
        return 0
    fi

    local om_dir="${MODEL_DIR}/om"
    mkdir -p "$om_dir"

    local model_type=$(cfg MODEL_TYPE "dynamic")

    # Append model type suffix to ATC output name to avoid collision
    # between static and dynamic models with the same MODEL_NAME.
    # Static:  model_static.om  (ATC does not add system suffix)
    # Dynamic: model_dynamic_linux_aarch64.om  (ATC adds system suffix)
    local atc_output_name="${MODEL_NAME}_${model_type}"
    local om_path="${om_dir}/${atc_output_name}.om"

    # Search for existing .om file
    # Static: only match exact filename (ATC does not add system suffix for static)
    # Dynamic: match exact first, then search for ATC-added system suffix
    local existing_om=""
    if [ -f "$om_path" ]; then
        existing_om="$om_path"
    elif [ "$model_type" = "dynamic" ]; then
        existing_om=$(find "$om_dir" -maxdepth 1 -name "${atc_output_name}*.om" -type f | head -1)
    fi

    if [ -n "$existing_om" ] && [ "$SKIP_ATC" = true ]; then
        log_info "OM model exists and --skip-atc specified, skipping"
        log_info "Using: $existing_om"
        CFG_OM_PATH="$existing_om"
        export CFG_OM_PATH
        return 0
    fi

    if [ ! -f "$model_path" ]; then
        log_error "Source model not found: $model_path"
        return 1
    fi

    if [ -n "$existing_om" ]; then
        log_info "OM model already exists: $existing_om"
        read -p "Re-convert? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping ATC conversion"
            CFG_OM_PATH="$existing_om"
            export CFG_OM_PATH
            return 0
        fi
    fi

    local framework=$(cfg FRAMEWORK 5)
    local soc_version=$(cfg SOC_VERSION "Ascend910_9382")
    local precision_mode=$(cfg ATC_PRECISION_MODE "force_fp16")
    local input_count=$(cfg INPUT_COUNT 1)
    local extra_args=$(cfg ATC_EXTRA_ARGS "")

    local input_shape_arg=""
    for ((i=0; i<input_count; i++)); do
        local name_key="INPUT_${i}_NAME"
        local shape_atc_key="INPUT_${i}_SHAPE_ATC"
        local shape_key="INPUT_${i}_SHAPE"

        local name=$(cfg "$name_key")
        local shape_atc=$(cfg "$shape_atc_key")
        local shape=$(cfg "$shape_key")

        local final_shape=""
        if [ "$model_type" = "dynamic" ]; then
            final_shape="${shape_atc:-$shape}"
        else
            final_shape="$shape"
        fi

        if [ -n "$input_shape_arg" ]; then
            input_shape_arg="${input_shape_arg};${name}:${final_shape}"
        else
            input_shape_arg="${name}:${final_shape}"
        fi
    done

    if [ "$model_type" = "static" ]; then
        log_info "Static model: using concrete shapes for --input_shape"
    fi

    local atc_cmd="atc"
    atc_cmd+=" --model=${model_path}"
    atc_cmd+=" --framework=${framework}"
    atc_cmd+=" --output=${om_dir}/${atc_output_name}"
    atc_cmd+=" --soc_version=${soc_version}"
    atc_cmd+=" --precision_mode=${precision_mode}"
    if [ -n "$input_shape_arg" ]; then
        atc_cmd+=" --input_shape=\"${input_shape_arg}\""
    fi
    if [ -n "$extra_args" ]; then
        atc_cmd+=" ${extra_args}"
    fi

    log_info "ATC command:"
    echo "  $atc_cmd"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would execute: $atc_cmd"
        CFG_OM_PATH="$om_path"
        export CFG_OM_PATH
        return 0
    fi

    eval "$atc_cmd"
    if [ $? -ne 0 ]; then
        log_error "ATC conversion failed"
        return 1
    fi

    # Detect the actual generated .om file
    # Static: ATC generates exact filename (e.g., model_static.om)
    # Dynamic: ATC may add system suffix (e.g., model_dynamic_linux_aarch64.om)
    local actual_om_path=""
    if [ -f "$om_path" ]; then
        actual_om_path="$om_path"
    elif [ "$model_type" = "dynamic" ]; then
        local found_om=$(find "$om_dir" -maxdepth 1 -name "${atc_output_name}*.om" -type f | head -1)
        if [ -n "$found_om" ]; then
            actual_om_path="$found_om"
            log_info "ATC generated file with suffix: $(basename "$found_om")"
        fi
    fi

    if [ -z "$actual_om_path" ]; then
        log_error "ATC conversion completed but .om file not found in $om_dir"
        log_error "Expected: ${atc_output_name}.om"
        return 1
    fi

    log_info "ATC conversion complete: $actual_om_path"
    CFG_OM_PATH="$actual_om_path"
    export CFG_OM_PATH
}

# =============================================================================
# Step 3: Run inference
# =============================================================================
do_infer() {
    log_step "Step 3: Run inference"

    if [ ! -f "${INFER_BIN}" ]; then
        log_error "Inference binary not found: ${INFER_BIN}"
        log_error "Run without --skip-build or --infer-only first"
        return 1
    fi

    local om_path="${CFG_OM_PATH}"
    if [ ! -f "$om_path" ]; then
        log_error "OM model not found: $om_path"
        log_error "Run without --skip-atc first"
        return 1
    fi

    local input_count=$(cfg INPUT_COUNT 1)
    local output_dir="${MODEL_DIR}/output"
    mkdir -p "$output_dir"

    local infer_args=()
    infer_args+=(--model "$om_path")
    infer_args+=(--output_dir "$output_dir")
    infer_args+=(--device_id "$DEVICE_ID")

    if [ "$USE_ASYNC" = true ]; then
        infer_args+=(--async)
    fi

    local model_type=$(cfg MODEL_TYPE "dynamic")
    if [ "$model_type" = "static" ]; then
        infer_args+=(--static)
        log_info "Static model mode: will skip SetDynamicInputTensorDesc"
    fi

    for ((i=0; i<input_count; i++)); do
        local name=$(cfg "INPUT_${i}_NAME")
        local shape=$(cfg "INPUT_${i}_SHAPE")
        local dtype=$(cfg "INPUT_${i}_DTYPE")
        local format=$(cfg "INPUT_${i}_FORMAT")
        local file=$(resolve_path "$(cfg "INPUT_${i}_FILE")")

        if [ ! -f "$file" ]; then
            log_error "Input data file not found: $file"
            return 1
        fi

        infer_args+=(--input "${name}:${shape}:${dtype}:${format}:${file}")
    done

    if [ "$DUMP_ENABLED" = true ]; then
        infer_args+=(--dump)

        local dump_path="${DUMP_PATH_OVERRIDE:-${MODEL_DIR}/dump_data}"
        mkdir -p "$dump_path"
        infer_args+=(--dump_path "$dump_path")

        local dump_mode="${DUMP_MODE:-$(cfg DUMP_MODE "output")}"
        infer_args+=(--dump_mode "$dump_mode")

        local dump_level="${DUMP_LEVEL:-$(cfg DUMP_LEVEL "op")}"
        infer_args+=(--dump_level "$dump_level")

        local dump_data="tensor"
        if [ "$DUMP_STATS" = true ]; then
            dump_data="stats"
        else
            dump_data=$(cfg DUMP_DATA "tensor")
        fi
        infer_args+=(--dump_data "$dump_data")

        local dump_layers="${DUMP_LAYERS:-$(cfg DUMP_LAYERS "")}"
        if [ -n "$dump_layers" ]; then
            infer_args+=(--dump_layer "$dump_layers")
        fi

        # Extract model name from actual .om filename (without extension)
        # This ensures model_name in acl.json matches what CANN expects
        local om_basename=$(basename "$om_path" .om)
        infer_args+=(--dump_model_name "$om_basename")
        log_info "Dump model_name: $om_basename (from .om filename)"
    fi

    if [ "$PROFILING_ENABLED" = true ]; then
        infer_args+=(--profiling)

        local profiling_path="${MODEL_DIR}/profiling_data"
        mkdir -p "$profiling_path"
        infer_args+=(--profiling_output "$profiling_path")
        
        log_info "Profiling enabled, data will be saved to: $profiling_path"
    fi

    if [ "$WARMUP_RUNS" -gt 0 ]; then
        infer_args+=(--warmup "$WARMUP_RUNS")
    fi

    if [ "$BENCH_RUNS" -gt 1 ]; then
        infer_args+=(--bench "$BENCH_RUNS")
    fi

    log_info "Inference command:"
    echo "  ${INFER_BIN} ${infer_args[*]}"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would execute above command"
    else
        "${INFER_BIN}" "${infer_args[@]}"
        log_info "Inference complete. Output: ${output_dir}/"
    fi
}

# =============================================================================
# Parse CLI arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-atc)
            SKIP_ATC=true
            shift
            ;;
        --atc-only)
            ATC_ONLY=true
            shift
            ;;
        --infer-only)
            INFER_ONLY=true
            shift
            ;;
        --dump)
            DUMP_ENABLED=true
            shift
            ;;
        --dump-stats)
            DUMP_ENABLED=true
            DUMP_STATS=true
            shift
            ;;
        --dump-mode)
            DUMP_MODE="$2"
            DUMP_ENABLED=true
            shift 2
            ;;
        --dump-level)
            DUMP_LEVEL="$2"
            DUMP_ENABLED=true
            shift 2
            ;;
        --dump-layers)
            DUMP_LAYERS="$2"
            DUMP_ENABLED=true
            shift 2
            ;;
        --dump-path)
            DUMP_PATH_OVERRIDE="$2"
            DUMP_ENABLED=true
            shift 2
            ;;
        --profiling)
            PROFILING_ENABLED=true
            shift
            ;;
        --async)
            USE_ASYNC=true
            shift
            ;;
        --device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        --list)
            LIST_MODELS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --warmup)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --bench)
            BENCH_RUNS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# =============================================================================
# Main
# =============================================================================

if [ "$LIST_MODELS" = true ]; then
    list_models
    exit 0
fi

if [ -z "$MODEL_NAME" ]; then
    log_error "Model name is required. Use -m <name> or --list to see available models."
    usage
    exit 1
fi

MODEL_DIR="${MODELS_DIR}/${MODEL_NAME}"
CONFIG_FILE="${MODEL_DIR}/config"

if [ ! -d "$MODEL_DIR" ]; then
    log_error "Model directory not found: ${MODEL_DIR}"
    echo ""
    echo "Available models:"
    list_models
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file not found: ${CONFIG_FILE}"
    echo ""
    echo "Create one from template:"
    echo "  cp models/.template/config ${MODEL_DIR}/config"
    exit 1
fi

# Save CLI dump overrides before load_config (which may overwrite them)
CLI_DUMP_ENABLED="$DUMP_ENABLED"
CLI_DUMP_STATS="$DUMP_STATS"
CLI_DUMP_MODE="$DUMP_MODE"
CLI_DUMP_LEVEL="$DUMP_LEVEL"
CLI_DUMP_LAYERS="$DUMP_LAYERS"
CLI_DUMP_PATH_OVERRIDE="$DUMP_PATH_OVERRIDE"

load_config "$CONFIG_FILE"

# Restore CLI dump overrides (load_config may have overwritten them from config file)
[ -n "$CLI_DUMP_MODE" ] && DUMP_MODE="$CLI_DUMP_MODE"
[ -n "$CLI_DUMP_LEVEL" ] && DUMP_LEVEL="$CLI_DUMP_LEVEL"
[ -n "$CLI_DUMP_LAYERS" ] && DUMP_LAYERS="$CLI_DUMP_LAYERS"
[ -n "$CLI_DUMP_PATH_OVERRIDE" ] && DUMP_PATH_OVERRIDE="$CLI_DUMP_PATH_OVERRIDE"
[ "$CLI_DUMP_STATS" = true ] && DUMP_STATS=true
[ "$CLI_DUMP_ENABLED" = true ] && DUMP_ENABLED=true

MODEL_NAME_CFG=$(cfg MODEL_NAME)
if [ -n "$MODEL_NAME_CFG" ]; then
    MODEL_NAME="$MODEL_NAME_CFG"
fi

INPUT_COUNT=$(cfg INPUT_COUNT 1)
MODEL_TYPE=$(cfg MODEL_TYPE "dynamic")

echo ""
echo "=========================================="
echo " Model: ${MODEL_NAME}"
echo " Type:  ${MODEL_TYPE}"
echo " Config: ${CONFIG_FILE}"
echo " Inputs: ${INPUT_COUNT}"
echo "=========================================="
for ((i=0; i<INPUT_COUNT; i++)); do
    name=$(cfg "INPUT_${i}_NAME")
    shape=$(cfg "INPUT_${i}_SHAPE")
    dtype=$(cfg "INPUT_${i}_DTYPE")
    format=$(cfg "INPUT_${i}_FORMAT")
    file=$(cfg "INPUT_${i}_FILE")
    echo "  Input[$i]: ${name} ${shape} ${dtype} ${format} -> ${file}"
done
echo "=========================================="
echo ""

if [ "$INFER_ONLY" = true ]; then
    SKIP_BUILD=true
    SKIP_ATC=true
fi

if [ "$ATC_ONLY" = true ]; then
    do_atc
    log_info "ATC-only mode complete"
    exit 0
fi

if [ "$SKIP_BUILD" = false ]; then
    do_build
fi

do_atc

do_infer

log_info "All steps completed successfully"
