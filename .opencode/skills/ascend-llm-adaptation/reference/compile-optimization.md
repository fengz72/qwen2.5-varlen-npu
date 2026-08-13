# 编译期优化

## NZ Weight Pass + MatMulV3 替换

### 原理

Ascend Cube 单元以 FRACTAL_NZ `[N1, K1, 16, 16]` 分块处理数据。
ND `[K, N]` 格式的权重每次访问都会产生运行时 TransData。在编译期转换
（通过常量折叠）可消除此开销。

### Pass 实现（C++ GE graph pass）

```cpp
#include "register/register_custom_pass.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "graph/ascend_string.h"
#include "ops_proto_nn.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>

using namespace ge;

namespace {
constexpr const char *kOpTypeMatMul = "MatMul";
constexpr const char *kOpTypeMatMulV2 = "MatMulV2";
constexpr const char *kOpTypeConst = "Const";
constexpr const char *kOpTypeConstant = "Constant";

static bool IsMatMulNode(GNode &node) {
    AscendString node_type;
    node.GetType(node_type);
    return node_type == kOpTypeMatMul || node_type == kOpTypeMatMulV2;
}

static bool IsConstNode(const GNodePtr &node) {
    AscendString weight_type;
    node->GetType(weight_type);
    return weight_type == kOpTypeConst || weight_type == kOpTypeConstant;
}

static bool ValidateWeightShape(const Shape &shape, int64_t &k_dim, int64_t &n_dim) {
    if (shape.GetDimNum() != 2) return false;
    k_dim = shape.GetDim(0);
    n_dim = shape.GetDim(1);
    return true;
}

static bool ReadConstTensor(const GNodePtr &weight_node, Tensor &const_tensor,
                             DataType &dt, int64_t &elem_size, const uint8_t *&src) {
    if (weight_node->GetAttr(AscendString("value"), const_tensor) != GRAPH_SUCCESS) return false;
    dt = const_tensor.GetTensorDesc().GetDataType();
    elem_size = GetSizeByDataType(dt);
    if (elem_size <= 0) return false;
    src = const_tensor.GetData();
    return src != nullptr;
}

static std::vector<uint8_t> ConvertNdToFractalNz(const uint8_t *src,
                                                   int64_t k_dim, int64_t n_dim, int64_t elem_size) {
    int64_t K1 = (k_dim + 15) / 16;
    int64_t N1 = (n_dim + 15) / 16;
    size_t nz_size = static_cast<size_t>(N1 * K1 * 16 * 16) * static_cast<size_t>(elem_size);
    std::vector<uint8_t> nz_data(nz_size, 0);
    for (int64_t n1 = 0; n1 < N1; n1++) {
        for (int64_t k1 = 0; k1 < K1; k1++) {
            for (int64_t r = 0; r < 16; r++) {
                int64_t sk = k1 * 16 + r;
                if (sk >= k_dim) break;
                for (int64_t c = 0; c < 16; c++) {
                    int64_t sn = n1 * 16 + c;
                    if (sn >= n_dim) break;
                    size_t src_off = static_cast<size_t>(sk * n_dim + sn) * static_cast<size_t>(elem_size);
                    size_t dst_off = static_cast<size_t>(((n1 * K1 + k1) * 16 + r) * 16 + c) * static_cast<size_t>(elem_size);
                    std::memcpy(&nz_data[dst_off], src + src_off, static_cast<size_t>(elem_size));
                }
            }
        }
    }
    return nz_data;
}

static TensorDesc BuildNzTensorDesc(const Shape &origin_shape, const Shape &nz_shape,
                                     DataType dt, size_t nz_size) {
    TensorDesc nz_desc(nz_shape, FORMAT_FRACTAL_NZ, dt);
    nz_desc.SetOriginShape(origin_shape);
    nz_desc.SetOriginFormat(FORMAT_ND);
    nz_desc.SetRealDimCnt(4);
    nz_desc.SetSize(nz_size);
    nz_desc.SetPlacement(kPlacementDevice);
    return nz_desc;
}

static void UpdateWeightNode(const GNodePtr &weight_node, const TensorDesc &nz_desc,
                              const std::vector<uint8_t> &nz_data) {
    Tensor new_tensor(nz_desc);
    new_tensor.SetData(nz_data.data(), nz_data.size());
    weight_node->SetAttr(AscendString("value"), new_tensor);
    weight_node->UpdateOutputDesc(0, nz_desc);
}

static Status ReplaceWithMatMulV3(GraphPtr &graph, GNode &node,
                                  const GNodePtr &weight_node, int32_t weight_index,
                                  const Shape &weight_shape, const Shape &nz_shape,
                                  DataType dt, size_t nz_size) {
    AscendString node_name;
    node.GetName(node_name);

    bool transpose_x1 = false, transpose_x2 = false;
    int64_t offset_x = 0;
    node.GetAttr(AscendString("transpose_x1"), transpose_x1);
    node.GetAttr(AscendString("transpose_x2"), transpose_x2);
    node.GetAttr(AscendString("offset_x"), offset_x);

    auto [x1_node, x1_idx] = node.GetInDataNodesAndPortIndexs(0);
    GNodePtr bias_node = nullptr;
    int32_t bias_idx = 0;
    auto [bias_in, bias_in_idx] = node.GetInDataNodesAndPortIndexs(2);
    if (bias_in) { bias_node = bias_in; bias_idx = bias_in_idx; }
    GNodePtr ow_node = nullptr;
    int32_t ow_idx = 0;
    auto [ow_in, ow_in_idx] = node.GetInDataNodesAndPortIndexs(3);
    if (ow_in) { ow_node = ow_in; ow_idx = ow_in_idx; }
    auto out_consumers = node.GetOutDataNodesAndPortIndexs(0);

    std::string v3_name = std::string(node_name.GetString()) + "_v3";
    op::MatMulV3 mm_v3_op(v3_name.c_str());
    mm_v3_op.set_attr_transpose_x1(transpose_x1);
    mm_v3_op.set_attr_transpose_x2(transpose_x2);
    mm_v3_op.set_attr_offset_x(offset_x);
    mm_v3_op.set_attr_opImplMode((int64_t)1);

    TensorDesc x1_desc;
    node.GetInputDesc(0, x1_desc);
    mm_v3_op.update_input_desc_x1(x1_desc);

    TensorDesc x2_nz_desc = BuildNzTensorDesc(weight_shape, nz_shape, dt, nz_size);
    mm_v3_op.update_input_desc_x2(x2_nz_desc);

    TensorDesc y_desc;
    node.GetOutputDesc(0, y_desc);
    mm_v3_op.update_output_desc_y(y_desc);

    if (bias_node) {
        TensorDesc bias_desc;
        node.GetInputDesc(2, bias_desc);
        mm_v3_op.update_input_desc_bias(bias_desc);
    }
    if (ow_node) {
        TensorDesc ow_desc;
        node.GetInputDesc(3, ow_desc);
        mm_v3_op.update_input_desc_offset_w(ow_desc);
    }

    auto mm_v3_node = graph->AddNodeByOp(mm_v3_op);

    // 手动补齐 GE 内部格式属性
    int64_t fmt_nd = 2;
    int64_t fmt_nz = 29;
    mm_v3_node.SetAttr(AscendString("input_desc_attr_format_for_int:0"), fmt_nd);
    mm_v3_node.SetAttr(AscendString("input_desc_attr_format_for_int:1"), fmt_nz);
    mm_v3_node.SetAttr(AscendString("input_desc_attr_origin_format_for_int:0"), fmt_nd);
    mm_v3_node.SetAttr(AscendString("input_desc_attr_origin_format_for_int:1"), fmt_nd);
    mm_v3_node.SetAttr(AscendString("output_desc_attr_format_for_int:0"), fmt_nd);
    mm_v3_node.SetAttr(AscendString("output_desc_attr_origin_format_for_int:0"), fmt_nd);

    AscendString cube_vector_core_type("MIX_AIC");
    mm_v3_node.SetAttr(AscendString("_cube_vector_core_type"), cube_vector_core_type);
    mm_v3_node.SetAttr(AscendString("_sgt_cube_vector_core_type"), cube_vector_core_type);

    // 断开旧边，重连到新节点
    graph->RemoveEdge(*x1_node, x1_idx, node, 0);
    graph->RemoveEdge(*weight_node, weight_index, node, 1);
    if (bias_node) { graph->RemoveEdge(*bias_node, bias_idx, node, 2); }
    if (ow_node) { graph->RemoveEdge(*ow_node, ow_idx, node, 3); }
    for (auto &consumer : out_consumers) {
        graph->RemoveEdge(node, 0, *consumer.first, consumer.second);
    }

    graph->AddDataEdge(*x1_node, x1_idx, mm_v3_node, 0);
    graph->AddDataEdge(*weight_node, weight_index, mm_v3_node, 1);
    if (bias_node) { graph->AddDataEdge(*bias_node, bias_idx, mm_v3_node, 2); }
    if (ow_node) { graph->AddDataEdge(*ow_node, ow_idx, mm_v3_node, 3); }
    for (auto &consumer : out_consumers) {
        graph->AddDataEdge(mm_v3_node, 0, *consumer.first, consumer.second);
    }

    graph->RemoveNode(node);
    return SUCCESS;
}

static Status MatMulWeightNZPass(GraphPtr &graph, CustomPassContext &ctx) {
    auto nodes = graph->GetAllNodes();
    int matmul_match = 0;
    int matmul_effect = 0;
    for (auto &node : nodes) {
        if (!IsMatMulNode(node)) continue;
        matmul_match++;

        auto [weight_node, weight_index] = node.GetInDataNodesAndPortIndexs(1);
        if (!IsConstNode(weight_node)) continue;

        TensorDesc weight_desc;
        if (node.GetInputDesc(1, weight_desc) != GRAPH_SUCCESS) continue;
        Shape weight_shape = weight_desc.GetShape();
        int64_t k_dim = 0, n_dim = 0;
        if (!ValidateWeightShape(weight_shape, k_dim, n_dim)) continue;

        Tensor const_tensor;
        DataType dt;
        int64_t elem_size = 0;
        const uint8_t *src = nullptr;
        if (!ReadConstTensor(weight_node, const_tensor, dt, elem_size, src)) continue;

        auto nz_data = ConvertNdToFractalNz(src, k_dim, n_dim, elem_size);
        size_t nz_size = nz_data.size();

        int64_t K1 = (k_dim + 15) / 16;
        int64_t N1 = (n_dim + 15) / 16;
        Shape nz_shape({N1, K1, 16, 16});

        TensorDesc nz_desc = BuildNzTensorDesc(weight_shape, nz_shape, dt, nz_size);
        UpdateWeightNode(weight_node, nz_desc, nz_data);

        ReplaceWithMatMulV3(graph, node, weight_node, weight_index,
                            weight_shape, nz_shape, dt, nz_size);

        matmul_effect++;
    }
    return SUCCESS;
}
}

REGISTER_CUSTOM_PASS("MatMulWeightNZPass")
    .CustomPassFn(MatMulWeightNZPass)
    .Stage(CustomPassStage::kAfterOriginGraphOptimize);
```

### MatMulV3 替换细节

手动创建 MatMulV3 节点时必须设置：
- `input_desc_attr_format_for_int:1 = 29`（NZ 格式码）
- `input_desc_attr_origin_format_for_int:0/1 = 2`（ND 原始格式）
- `output_desc_attr_format_for_int:0 = 2`（ND 输出格式）
- `_cube_vector_core_type = "MIX_AIC"`（防止错误核调度）
- `_sgt_cube_vector_core_type = "MIX_AIC"`
- 重连所有输入/输出边（bias、offset_w、消费者）
- 重连后删除旧节点

### 编译和安装

```bash
g++ -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -o libnz_weight_pass.so \
    nz_weight_pass.cpp \
    -I${ASCEND_HOME_PATH}/include \
    -I${ASCEND_HOME_PATH}/opp/built-in/op_graph/inc \
    -L${ASCEND_HOME_PATH}/lib64 \
    -L${ASCEND_HOME_PATH}/opp/built-in/op_graph/lib/linux/$(uname -m) \
    -lgraph -lregister

mkdir -p ${ASCEND_HOME_PATH}/opp/vendors/custom_nz_pass/custom_fusion_passes
cp libnz_weight_pass.so ${ASCEND_HOME_PATH}/opp/vendors/custom_nz_pass/custom_fusion_passes/
```

安装后重新执行 ATC 编译，pass 自动生效。

### 适用条件

- 权重内轴（N）≥ 65536（足够大才值得 NZ 分块）
- 权重是 Const 节点（frozen_parameter=1）
- MatMul 是计算瓶颈（非访存瓶颈）

## AICore 限核

用于多模型共享同一张 NPU 卡的场景：

```bash
# 单个整数 N：AIC=N, AIV=N*2（1:2 比例）
atc ... --aicore_num="12"          # → 12 AIC, 24 AIV

# 显式 aic|aiv 格式
atc ... --aicore_num="12|24"
```

前置条件：安装 GE Compiler 和 GE Executor：

```bash
./cann-ge-compiler_linux-aarch64.run --full -q
./cann-ge-executor_linux-aarch64.run --full -q
```

OM 文件名加 `_c{aic}_{aiv}` 后缀。用不同核数 benchmark 找最优分配。

## 其他编译选项

| 选项 | 标志 | 用途 |
|------|------|------|
| 调试日志 | `--log=debug` | 诊断 ATC 失败 |
| GE 图 dump | `DUMP_GRAPH_PATH=... DUMP_GE_GRAPH=2` | 检查图变换过程 |
| 精度模式 | （默认 force_fp16） | 对 fp16 模型已最优 |
