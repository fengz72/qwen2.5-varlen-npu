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
    if (shape.GetDimNum() != 2) {
        std::cout << "the weight shape is not 2" << std::endl;
        return false;
    }
    k_dim = shape.GetDim(0);
    n_dim = shape.GetDim(1);
    return true;
}

static bool ReadConstTensor(const GNodePtr &weight_node, Tensor &const_tensor,
                            DataType &dt, int64_t &elem_size, const uint8_t *&src) {
    if (weight_node->GetAttr(AscendString("value"), const_tensor) != GRAPH_SUCCESS) {
        std::cout << "failed to read const value" << std::endl;
        return false;
    }
    dt = const_tensor.GetTensorDesc().GetDataType();
    elem_size = GetSizeByDataType(dt);
    if (elem_size <= 0) {
        std::cout << "unsupported dtype" << std::endl;
        return false;
    }
    src = const_tensor.GetData();
    if (src == nullptr) {
        std::cout << "const data is null" << std::endl;
        return false;
    }
    return true;
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

// 用MatMulV3替换原MatMul/MatMulV2节点：
// 1. 继承原节点的属性和输入输出连接关系
// 2. x2输入desc设为FRACTAL_NZ格式以匹配已重排的权重
// 3. 补齐GE内部格式属性，重连边并删除旧节点
static Status ReplaceWithMatMulV3(GraphPtr &graph, GNode &node,
                                 const GNodePtr &weight_node, int32_t weight_index,
                                 const Shape &weight_shape, const Shape &nz_shape,
                                 DataType dt, size_t nz_size) {
    AscendString node_name;
    node.GetName(node_name);

    // 1. 继承原节点的transpose/offset_x等属性
    bool transpose_x1 = false, transpose_x2 = false;
    int64_t offset_x = 0;
    node.GetAttr(AscendString("transpose_x1"), transpose_x1);
    node.GetAttr(AscendString("transpose_x2"), transpose_x2);
    node.GetAttr(AscendString("offset_x"), offset_x);

    // 2. 收集原节点所有输入(x1/weight/bias/offset_w)及输出消费者，后续重连边时使用
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

    // 3. 创建MatMulV3算子并设置属性(opImplMode=1表示权重为NZ格式)
    std::string v3_name = std::string(node_name.GetString()) + "_v3";
    op::MatMulV3 mm_v3_op(v3_name.c_str());
    mm_v3_op.set_attr_transpose_x1(transpose_x1);
    mm_v3_op.set_attr_transpose_x2(transpose_x2);
    mm_v3_op.set_attr_offset_x(offset_x);
    mm_v3_op.set_attr_opImplMode((int64_t)1);

    // 4. 设置输入输出desc：x1沿用原ND格式，x2设为NZ格式，y沿用原输出desc
    TensorDesc x1_desc;
    node.GetInputDesc(0, x1_desc);
    std::vector<std::pair<int64_t, int64_t>> x1_range;
    x1_desc.GetShapeRange(x1_range);
    mm_v3_op.update_input_desc_x1(x1_desc);

    TensorDesc x2_nz_desc = BuildNzTensorDesc(weight_shape, nz_shape, dt, nz_size);
    mm_v3_op.update_input_desc_x2(x2_nz_desc);

    TensorDesc y_desc;
    node.GetOutputDesc(0, y_desc);
    std::vector<std::pair<int64_t, int64_t>> y_range;
    y_desc.GetShapeRange(y_range);
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

    // 5. 补齐GE内部desc格式属性：x1输入ND、x2输入NZ、输出ND
    //    旧节点由GE优化pass自动设置，新节点需手动补
    int64_t fmt_nd = 2;
    int64_t fmt_nz = 29;
    mm_v3_node.SetAttr(AscendString("input_desc_attr_format_for_int:0"), fmt_nd);
    mm_v3_node.SetAttr(AscendString("input_desc_attr_format_for_int:1"), fmt_nz);
    mm_v3_node.SetAttr(AscendString("input_desc_attr_origin_format_for_int:0"), fmt_nd);
    mm_v3_node.SetAttr(AscendString("input_desc_attr_origin_format_for_int:1"), fmt_nd);
    mm_v3_node.SetAttr(AscendString("output_desc_attr_format_for_int:0"), fmt_nd);
    mm_v3_node.SetAttr(AscendString("output_desc_attr_origin_format_for_int:0"), fmt_nd);

    // 补齐核类型属性：GE后续pass会将MatMulV3的_cube_vector_core_type推断为"MIX"，
    // 而MatMul应为"AiCore"，需从原节点继承避免算子调度到错误核类型
    AscendString cube_vector_core_type("MIX_AIC");
    mm_v3_node.SetAttr(AscendString("_cube_vector_core_type"), cube_vector_core_type);
    mm_v3_node.SetAttr(AscendString("_sgt_cube_vector_core_type"), cube_vector_core_type);

    // 6. 断开旧节点所有边，按原拓扑关系重连到新MatMulV3节点
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

    // 7. 删除旧MatMul/MatMulV2节点
    graph->RemoveNode(node);
    return SUCCESS;
}

// MatMul权重NZ格式转换pass主流程：
// 遍历图中所有MatMul/MatMulV2节点，将常量权重从ND格式重排为FRACTAL_NZ格式，
// 并将原节点替换为原生支持NZ权重的MatMulV3节点，以提升Cube单元访存效率。
static Status MatMulWeightNZPass(GraphPtr &graph, CustomPassContext &ctx) {
    auto nodes = graph->GetAllNodes();
    int matmul_match = 0;
    int matmul_effect = 0;
    for (auto &node : nodes) {
        if (!IsMatMulNode(node)) {
            continue;
        }
        matmul_match++;

        // 1. 取权重输入(第1路)，必须是Const/Constant常量节点
        auto [weight_node, weight_index] = node.GetInDataNodesAndPortIndexs(1);
        if (!IsConstNode(weight_node)) {
            std::cout << "the weight is not const node" << std::endl;
            continue;
        }

        // 2. 校验权重shape：必须为2D且n_dim>=65536(内轴足够大才值得做NZ)
        TensorDesc weight_desc;
        if (node.GetInputDesc(1, weight_desc) != GRAPH_SUCCESS) {
            continue;
        }
        Shape weight_shape = weight_desc.GetShape();
        int64_t k_dim = 0, n_dim = 0;
        if (!ValidateWeightShape(weight_shape, k_dim, n_dim)) {
            continue;
        }

        // 3. 读取常量权重数据及dtype
        Tensor const_tensor;
        DataType dt;
        int64_t elem_size = 0;
        const uint8_t *src = nullptr;
        if (!ReadConstTensor(weight_node, const_tensor, dt, elem_size, src)) {
            continue;
        }

        // 4. 将权重数据从ND [K,N]重排为FRACTAL_NZ [N1,K1,16,16]
        auto nz_data = ConvertNdToFractalNz(src, k_dim, n_dim, elem_size);
        size_t nz_size = nz_data.size();

        int64_t K1 = (k_dim + 15) / 16;
        int64_t N1 = (n_dim + 15) / 16;
        Shape nz_shape({N1, K1, 16, 16});

        // 5. 构建NZ TensorDesc并回写权重数据，更新权重节点输出desc
        TensorDesc nz_desc = BuildNzTensorDesc(weight_shape, nz_shape, dt, nz_size);
        UpdateWeightNode(weight_node, nz_desc, nz_data);

        // 6. 创建MatMulV3替换原MatMul/MatMulV2，重连输入输出边并删除旧节点
        ReplaceWithMatMulV3(graph, node, weight_node, weight_index,
                            weight_shape, nz_shape, dt, nz_size);

        matmul_effect++;
    }
    std::cout << "the matmul_match num is  " << matmul_match << std::endl;
    std::cout << "the matmul_effect num is  " << matmul_effect << std::endl;
    return SUCCESS;
}
}

REGISTER_CUSTOM_PASS("MatMulWeightNZPass")
    .CustomPassFn(MatMulWeightNZPass)
    .Stage(CustomPassStage::kAfterOriginGraphOptimize);
