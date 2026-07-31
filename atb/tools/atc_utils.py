"""
ATC 编译工具 — 将 AIR 模型编译为 OM
"""

import os
import glob
import subprocess

import numpy


def run_atc(air_path, om_dir, soc, input_shape=None, is_debug=False, aicore_num=None):
    """执行 ATC 命令将 AIR 编译为 OM。

    --framework=1 表示输入为 AIR 格式 (GE 原生图格式)。
    OM 输出到 om_dir 目录, 文件名与 AIR 相同。
    使用自定义 Pass (NzWeightPass) 在 Const→MatMul 间插入 TransData,
    利用常量折叠在编译期完成大权重 ND→FRACTAL_NZ 转换, 消除运行时 TransData。
    NzWeightPass 已安装到 CANN opp/vendors/custom_nz_pass/custom_fusion_passes/。

    aicore_num: 限制目标设备运行时使用的 AICore 数量, None 则用硬件默认值。
                指定时 OM 文件名追加 _aicore{N} 后缀, 避免覆盖。
    """
    os.makedirs(om_dir, exist_ok=True)
    air_basename = os.path.splitext(os.path.basename(air_path))[0]
    if aicore_num:
        om_name = f"{air_basename}_aicore{aicore_num}"
    else:
        om_name = air_basename
    om_output = os.path.join(om_dir, om_name)

    cmd = (
        f"atc --framework=1"
        f" --model={air_path}"
        f" --output={om_output}"
        f" --soc_version={soc}"
    )
    if input_shape:
        cmd += f' --input_shape="{input_shape}"'

    if aicore_num:
        cmd += f' --aicore_num={aicore_num}'

    if is_debug:
        cmd += f' --log=debug'

    print(f"=== 执行 ATC 编译 ===")
    print(f"  命令: {cmd}\n")

    env = os.environ.copy()
    numpy_site = os.path.dirname(os.path.dirname(numpy.__file__))
    env['PYTHONPATH'] = f"{numpy_site}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] ATC 编译失败:")
        print(result.stderr[-3000:])
        return None

    om_file = om_output + ".om"
    if not os.path.exists(om_file):
        candidates = glob.glob(f"{om_output}*.om")
        if candidates:
            om_file = candidates[0]
        else:
            print(f"[ERROR] OM 文件未生成: {om_file}")
            return None

    file_size = os.path.getsize(om_file) / 1024 / 1024
    print(f"=== OM 编译完成: {om_file} ({file_size:.1f} MB) ===\n")
    return om_file
