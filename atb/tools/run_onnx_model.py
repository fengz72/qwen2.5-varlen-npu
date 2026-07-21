"""
运行动态 batch ONNX 模型（输入为 .npy 文件）

用法示例:
    # 单输入模型，单个 .npy 文件（数组或字典）
    python run_onnx_npy.py --model model.onnx --inputs input.npy

    # 多输入模型，指定每个输入的名称
    python run_onnx_npy.py --model model.onnx --inputs img.npy mask.npy --input_names images masks

    # 基准测试（运行 100 次）
    python run_onnx_npy.py --model model.onnx --inputs input.npy --benchmark 100
"""

import argparse
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description="使用 .npy 文件推理动态 batch ONNX 模型")
    parser.add_argument("--model", required=True, help="ONNX 模型文件路径")
    parser.add_argument("--inputs", nargs="+", required=True, help="一个或多个 .npy 输入文件")
    parser.add_argument("--input_names", nargs="+", default=None,
                        help="模型输入节点名称（数量需与输入文件一致，不提供则自动获取）")
    parser.add_argument("--output_names", nargs="+", default=None, help="期望的输出节点名称（不提供则输出全部）")
    parser.add_argument("--benchmark", type=int, default=0, help="基准测试迭代次数（0 表示禁用）")
    args = parser.parse_args()

    # 优先创建会话，以便获取模型输入输出信息
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)
    print(f"使用的执行提供者: {session.get_providers()}")

    model_inputs = session.get_inputs()
    model_outputs = session.get_outputs()

    # 加载输入数据并构建 feed_dict
    if len(args.inputs) == 1 and args.input_names is None:
        # 单个 .npy 文件，尝试加载为数组或字典
        data = np.load(args.inputs[0], allow_pickle=True)
        if isinstance(data, np.ndarray):
            # 单个数组，默认使用模型的第一个输入名称
            input_name = model_inputs[0].name
            print(f"单数组输入，将自动映射到输入节点: '{input_name}'")
            feed_dict = {input_name: data}
        elif isinstance(data, dict) or (hasattr(data, 'item') and isinstance(data.item(), dict)):
            # .npy 中保存了字典 (例如用 np.save('data.npy', {'images': arr, ...}))
            feed_dict = data.item() if hasattr(data, 'item') else data
            print(f"从 .npy 加载了字典，包含键: {list(feed_dict.keys())}")
        else:
            raise ValueError("无法识别 .npy 文件内容。请提供数组或字典格式。")
    else:
        # 多个 .npy 文件，每个对应一个模型输入
        arrays = [np.load(f) for f in args.inputs]
        if args.input_names:
            if len(args.input_names) != len(arrays):
                raise ValueError(f"--input_names 数量 ({len(args.input_names)}) 必须与 --inputs 数量 ({len(arrays)}) 一致")
            input_names = args.input_names
        else:
            # 自动使用模型定义的输入名称列表
            if len(arrays) != len(model_inputs):
                raise ValueError(
                    f"输入文件数量 ({len(arrays)}) 与模型输入数量 ({len(model_inputs)}) 不匹配，"
                    "请使用 --input_names 明确指定名称。")
            input_names = [inp.name for inp in model_inputs]
        feed_dict = {name: arr for name, arr in zip(input_names, arrays)}
        print(f"构建的 feed_dict 键: {list(feed_dict.keys())}")

    # 打印输入形状，方便调试
    for name, arr in feed_dict.items():
        print(f"输入 '{name}': {arr.shape} ({arr.dtype})")

    # 确定输出名称
    if args.output_names:
        output_names = args.output_names
    else:
        output_names = [out.name for out in model_outputs]
        print(f"将输出所有节点: {output_names}")

    # 推理
    print("\n正在推理...")
    outputs = session.run(output_names, feed_dict)

    # 打印输出
    for name, out in zip(output_names, outputs):
        print(f"\n输出 '{name}': shape {out.shape}, dtype {out.dtype}")
        # 如果数组不太大，打印实际值；否则只打印摘要
        if out.size < 100:
            print(out)
        else:
            print(f"  前 5 个值: {out.flat[:5]}")

        np.save(name, out)

    # 基准测试
    if args.benchmark > 0:
        import time
        # 预热
        _ = session.run(output_names, feed_dict)
        start = time.perf_counter()
        for _ in range(args.benchmark):
            session.run(output_names, feed_dict)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / args.benchmark) * 1000
        print(f"\n基准测试: {args.benchmark} 次运行，平均 {avg_ms:.2f} ms/次")


if __name__ == "__main__":
    main()
