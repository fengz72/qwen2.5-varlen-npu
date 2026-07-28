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

static Status MatMulWeightNZPass(GraphPtr &graph, CustomPassContext &ctx) {
    auto nodes = graph->GetAllNodes();
    int matmul_match = 0;
    int matmul_effect = 0;
    for (auto &node : nodes) {
        AscendString node_type;
        node.GetType(node_type);
        if (node_type != kOpTypeMatMul && node_type != kOpTypeMatMulV2) {
            continue;
        }
        matmul_match++;
        AscendString node_name;
        node.GetName(node_name);
        // std::cout << "the matmul name is : " << node_name << std::endl;
        // check the weigt
        auto [weight_node, weight_index] = node.GetInDataNodesAndPortIndexs(1);
        AscendString weight_type;
        weight_node->GetType(weight_type);
        // check the node is const
        if (weight_type != kOpTypeConst && weight_type != kOpTypeConstant) {
            std::cout << "the weight is not const node" << std::endl;
            continue;
        }
        // check the index dims
        TensorDesc weight_desc;
        if (node.GetInputDesc(1, weight_desc) != GRAPH_SUCCESS) {
            continue;
        }
        Shape weight_shape = weight_desc.GetShape();
        if (weight_shape.GetDimNum() != 2) {
            std::cout << "the weight shape is not 2" << std::endl;
            continue;
        }
        int64_t k_dim = weight_shape.GetDim(0);
        int64_t n_dim = weight_shape.GetDim(1);
        if (n_dim < 65536) {
            std::cout << "the weight inner axis is less 65536" << std::endl;
            continue;
        }
        // read const tensor data
        Tensor const_tensor;
        if (weight_node->GetAttr(AscendString("value"), const_tensor) != GRAPH_SUCCESS) {
            std::cout << "failed to read const value" << std::endl;
            continue;
        }
        DataType dt = const_tensor.GetTensorDesc().GetDataType();
        int64_t elem_size = GetSizeByDataType(dt);
        if (elem_size <= 0) {
            std::cout << "unsupported dtype" << std::endl;
            continue;
        }
        const uint8_t *src = const_tensor.GetData();
        if (src == nullptr) {
            std::cout << "const data is null" << std::endl;
            continue;
        }
        // rearrange ND [K,N] -> FRACTAL_NZ [N1,K1,16,16]
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
        // build nz tensor desc, keep origin shape/format for logical [K,N]
        Shape nz_shape({N1, K1, 16, 16});
        TensorDesc nz_desc(nz_shape, FORMAT_FRACTAL_NZ, dt);
        nz_desc.SetOriginShape(weight_shape);
        nz_desc.SetOriginFormat(FORMAT_ND);
        nz_desc.SetRealDimCnt(4);
        nz_desc.SetSize(nz_size);
        nz_desc.SetPlacement(kPlacementDevice);
        // write back const value and update output desc
        Tensor new_tensor(nz_desc);
        new_tensor.SetData(nz_data.data(), nz_size);
        weight_node->SetAttr(AscendString("value"), new_tensor);
        weight_node->UpdateOutputDesc(0, nz_desc);
        // create MatMulV3 to replace MatMulV2 (MatMulV3 natively supports NZ weight)
        bool transpose_x1 = false, transpose_x2 = false;
        int64_t offset_x = 0;
        node.GetAttr(AscendString("transpose_x1"), transpose_x1);
        node.GetAttr(AscendString("transpose_x2"), transpose_x2);
        node.GetAttr(AscendString("offset_x"), offset_x);
        // collect all input nodes before rewiring
        auto [x1_node, x1_idx] = node.GetInDataNodesAndPortIndexs(0);
        GNodePtr bias_node = nullptr;
        int32_t bias_idx = 0;
        auto [bias_in, bias_in_idx] = node.GetInDataNodesAndPortIndexs(2);
        if (bias_in) { bias_node = bias_in; bias_idx = bias_in_idx; }
        GNodePtr ow_node = nullptr;
        int32_t ow_idx = 0;
        auto [ow_in, ow_in_idx] = node.GetInDataNodesAndPortIndexs(3);
        if (ow_in) { ow_node = ow_in; ow_idx = ow_in_idx; }
        // collect output consumers
        auto out_consumers = node.GetOutDataNodesAndPortIndexs(0);
        // create MatMulV3 operator (natively supports NZ weight)
        std::string v3_name = std::string(node_name.GetString()) + "_v3";
        op::MatMulV3 mm_v3_op(v3_name.c_str());
        mm_v3_op.set_attr_transpose_x1(transpose_x1);
        mm_v3_op.set_attr_transpose_x2(transpose_x2);
        mm_v3_op.set_attr_offset_x(offset_x);
        mm_v3_op.set_attr_opImplMode((int64_t)1);
        // set input/output descs on operator before adding to graph
        TensorDesc x1_desc;
        node.GetInputDesc(0, x1_desc);
        std::vector<std::pair<int64_t, int64_t>> x1_range;
        x1_desc.GetShapeRange(x1_range);
        mm_v3_op.update_input_desc_x1(x1_desc);
        // build x2 nz desc with full metadata
        TensorDesc x2_nz_desc(nz_shape, FORMAT_FRACTAL_NZ, dt);
        x2_nz_desc.SetOriginShape(weight_shape);
        x2_nz_desc.SetOriginFormat(FORMAT_ND);
        x2_nz_desc.SetRealDimCnt(4);
        x2_nz_desc.SetSize(nz_size);
        x2_nz_desc.SetPlacement(kPlacementDevice);
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
        //补上 GE 内部 desc 属性，旧节点由 GE 优化 pass 自动设置，新节点需手动补
        int64_t fmt_nd = 2;
        int64_t fmt_nz = 29;
        mm_v3_node.SetAttr(AscendString("input_desc_attr_format_for_int:0"), fmt_nd);
        mm_v3_node.SetAttr(AscendString("input_desc_attr_format_for_int:1"), fmt_nz);
        mm_v3_node.SetAttr(AscendString("input_desc_attr_origin_format_for_int:0"), fmt_nd);
        mm_v3_node.SetAttr(AscendString("input_desc_attr_origin_format_for_int:1"), fmt_nd);
        mm_v3_node.SetAttr(AscendString("output_desc_attr_format_for_int:0"), fmt_nd);
        mm_v3_node.SetAttr(AscendString("output_desc_attr_origin_format_for_int:0"), fmt_nd);
        // remove old edges
        graph->RemoveEdge(*x1_node, x1_idx, node, 0);
        graph->RemoveEdge(*weight_node, weight_index, node, 1);
        if (bias_node) { graph->RemoveEdge(*bias_node, bias_idx, node, 2); }
        if (ow_node) { graph->RemoveEdge(*ow_node, ow_idx, node, 3); }
        for (auto &consumer : out_consumers) {
            graph->RemoveEdge(node, 0, *consumer.first, consumer.second);
        }
        // add new edges
        graph->AddDataEdge(*x1_node, x1_idx, mm_v3_node, 0);
        graph->AddDataEdge(*weight_node, weight_index, mm_v3_node, 1);
        if (bias_node) { graph->AddDataEdge(*bias_node, bias_idx, mm_v3_node, 2); }
        if (ow_node) { graph->AddDataEdge(*ow_node, ow_idx, mm_v3_node, 3); }
        for (auto &consumer : out_consumers) {
            graph->AddDataEdge(mm_v3_node, 0, *consumer.first, consumer.second);
        }
        // remove old MatMulV2 node
        graph->RemoveNode(node);
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
