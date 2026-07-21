#!/usr/bin/env python3
"""
Generate model config from ONNX model.

Usage:
    python3 tools/generate_config.py --onnx model.onnx --output models/bert --name bert
    python3 tools/generate_config.py --onnx model.onnx --output models/bert_static --name bert --static
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import onnx
    from onnx import TensorProto
except ImportError:
    print("[ERROR] onnx package not found. Install with: pip install onnx")
    sys.exit(1)


# ONNX dtype to ACL dtype mapping
ONNX_TO_ACL_DTYPE = {
    TensorProto.FLOAT: "float",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.STRING: "string",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "double",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
}


def get_dim_name(dim_idx, total_dims, input_name):
    """Generate intelligent variable name based on dimension position and context."""
    # Common patterns
    if dim_idx == 0:
        return "BATCH_SIZE"
    elif dim_idx == 1 and total_dims == 2:
        return "SEQ_LEN"
    elif dim_idx == 1 and total_dims >= 3:
        return "CHANNELS" if total_dims == 4 else "SEQ_LEN"
    elif dim_idx == 2 and total_dims == 4:
        return "HEIGHT"
    elif dim_idx == 3 and total_dims == 4:
        return "WIDTH"
    else:
        # Generic naming
        return f"DIM{dim_idx}"


def parse_onnx_inputs(model_path):
    """Parse ONNX model and extract input information."""
    model = onnx.load(model_path)
    inputs = []
    
    for idx, inp in enumerate(model.graph.input):
        # Skip initializers (model weights)
        if inp.name in [init.name for init in model.graph.initializer]:
            continue
        
        tensor_type = inp.type.tensor_type
        dtype = tensor_type.elem_type
        
        # Extract shape
        shape = []
        dynamic_dims = {}  # dim_idx -> symbolic name or None
        
        if tensor_type.HasField('shape'):
            for dim_idx, dim in enumerate(tensor_type.shape.dim):
                if dim.HasField('dim_value'):
                    if dim.dim_value == 0:
                        # Dynamic dimension (value=0 means unknown)
                        dynamic_dims[dim_idx] = None
                        shape.append(None)
                    else:
                        shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    # Dynamic dimension with symbolic name
                    dynamic_dims[dim_idx] = dim.dim_param
                    shape.append(None)
                else:
                    # Unknown dimension
                    dynamic_dims[dim_idx] = None
                    shape.append(None)
        
        inputs.append({
            'index': idx,
            'name': inp.name,
            'dtype': dtype,
            'shape': shape,
            'dynamic_dims': dynamic_dims,
            'is_dynamic': len(dynamic_dims) > 0
        })
    
    return inputs


def generate_config(inputs, model_name, model_file, static_mode=False):
    """Generate config file content."""
    lines = []
    
    # Header
    lines.append("# =============================================================================")
    lines.append(f"# Model: {model_name}")
    lines.append("# =============================================================================")
    lines.append("")
    
    # Collect all dynamic dimensions and generate variables
    # Group by ONNX symbolic name to share variables across inputs
    dim_vars = {}       # (input_idx, dim_idx) -> var_name
    var_values = {}     # var_name -> default_value
    sym_name_to_var = {}  # onnx_symbolic_name -> var_name (for cross-input sharing)
    
    for inp in inputs:
        total_dims = len(inp['shape'])
        for dim_idx, sym_name in inp['dynamic_dims'].items():
            if sym_name and sym_name in sym_name_to_var:
                # Same ONNX symbolic name as another input - reuse variable
                var_name = sym_name_to_var[sym_name]
            else:
                var_name = get_dim_name(dim_idx, total_dims, inp['name'])
                base_name = var_name
                counter = 2
                while var_name in var_values:
                    var_name = f"{base_name}_{counter}"
                    counter += 1
                var_values[var_name] = 1  # Default value
                if sym_name:
                    sym_name_to_var[sym_name] = var_name
            
            dim_vars[(inp['index'], dim_idx)] = var_name
    
    # Dimension variables section (for both static and dynamic modes)
    if var_values:
        if static_mode:
            lines.append("# ---- Dimension Variables (edit here to change all inputs) ----")
        else:
            lines.append("# ---- Dimension Variables (edit here to change runtime shape) ----")
            lines.append("# SHAPE: runtime shape passed to inference (edit these values)")
            lines.append("# SHAPE_ATC: shape range for ATC conversion (do NOT edit)")
        for var_name, default_val in var_values.items():
            lines.append(f"{var_name}={default_val}")
        lines.append("")
    
    # Basic settings
    lines.append("# ---- Basic Settings ----")
    lines.append(f"MODEL_NAME={model_name}")
    lines.append(f"MODEL_FILE={model_file}")
    lines.append("FRAMEWORK=5")
    lines.append("SOC_VERSION=Ascend910_9382")
    lines.append(f"MODEL_TYPE={'static' if static_mode else 'dynamic'}")
    lines.append("")
    
    # Input specification
    lines.append("# ---- Input Specification ----")
    lines.append(f"INPUT_COUNT={len(inputs)}")
    lines.append("")
    
    for inp in inputs:
        idx = inp['index']
        lines.append(f"INPUT_{idx}_NAME={inp['name']}")
        
        # Generate SHAPE: use variable references for dynamic dims
        shape_parts = []
        for dim_idx, dim_val in enumerate(inp['shape']):
            if dim_val is None:
                var_name = dim_vars[(idx, dim_idx)]
                shape_parts.append(f"${{{var_name}}}")
            else:
                shape_parts.append(str(dim_val))
        shape_str = ",".join(shape_parts)
        lines.append(f"INPUT_{idx}_SHAPE={shape_str}")
        
        # Dtype
        acl_dtype = ONNX_TO_ACL_DTYPE.get(inp['dtype'], "float")
        lines.append(f"INPUT_{idx}_DTYPE={acl_dtype}")
        
        # Format
        lines.append(f"INPUT_{idx}_FORMAT=ND")
        
        # File
        lines.append(f"INPUT_{idx}_FILE=input_data/{inp['name']}.bin")
        
        # SHAPE_ATC (only for dynamic mode): always use -1 for dynamic dims
        if not static_mode and inp['is_dynamic']:
            atc_parts = []
            for dim_val in inp['shape']:
                if dim_val is None:
                    atc_parts.append("-1")
                else:
                    atc_parts.append(str(dim_val))
            atc_str = ",".join(atc_parts)
            lines.append(f"INPUT_{idx}_SHAPE_ATC={atc_str}")
        
        lines.append("")
    
    # ATC and dump settings
    lines.append("ATC_PRECISION_MODE=force_fp16")
    lines.append("")
    lines.append("DUMP_MODE=output")
    lines.append("DUMP_LEVEL=op")
    lines.append("DUMP_DATA=tensor")
    lines.append("")
    
    return "\n".join(lines), var_values


def main():
    parser = argparse.ArgumentParser(description="Generate model config from ONNX model")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--name", required=True, help="Model name")
    parser.add_argument("--static", action="store_true", help="Generate static mode config")
    parser.add_argument("--create-inputs", action="store_true", 
                       help="Create empty input_data directory and placeholder files")
    
    args = parser.parse_args()
    
    # Check ONNX file exists
    if not os.path.exists(args.onnx):
        print(f"[ERROR] ONNX file not found: {args.onnx}")
        sys.exit(1)
    
    # Parse ONNX model
    print(f"[INFO] Parsing ONNX model: {args.onnx}")
    inputs = parse_onnx_inputs(args.onnx)
    
    if not inputs:
        print("[ERROR] No inputs found in ONNX model")
        sys.exit(1)
    
    print(f"[INFO] Found {len(inputs)} inputs:")
    for inp in inputs:
        dynamic_str = " (dynamic)" if inp['is_dynamic'] else ""
        print(f"  - {inp['name']}: {inp['shape']}{dynamic_str}")
    
    # Generate config
    model_file = f"onnx/{os.path.basename(args.onnx)}"
    config_content, var_values = generate_config(inputs, args.name, model_file, args.static)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write config file
    config_path = output_dir / "config"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"[INFO] Generated config: {config_path}")
    
    # Create onnx directory and copy/symlink model
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    
    onnx_dest = onnx_dir / os.path.basename(args.onnx)
    if not onnx_dest.exists():
        # Create symlink
        os.symlink(os.path.abspath(args.onnx), onnx_dest)
        print(f"[INFO] Created symlink: {onnx_dest}")
    
    # Create input_data directory if requested
    if args.create_inputs:
        input_dir = output_dir / "input_data"
        input_dir.mkdir(exist_ok=True)
        print(f"[INFO] Created input_data directory: {input_dir}")
        print("[INFO] Please create input .bin files in this directory")
    
    print("\n[SUCCESS] Config generation complete!")
    print(f"\nNext steps:")
    if var_values:
        print(f"  1. Edit dimension variables at the top of {config_path}:")
        for var_name, default_val in var_values.items():
            print(f"       {var_name}={default_val}  (change to your actual value)")
    else:
        print(f"  1. Review {config_path}")
    print(f"  2. Create input data files in {output_dir}/input_data/")
    print(f"  3. Run: ./run.sh -m {args.name}")


if __name__ == "__main__":
    main()
