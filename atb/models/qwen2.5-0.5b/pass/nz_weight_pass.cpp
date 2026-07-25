#include "register/register_custom_pass.h"
#include "graph/graph.h"
#include "transformation_ops.h" 
#include <iostream>
#include <string>

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
        // add transdata
        std::string transdata_name = std::string(node_name.GetString()) + "_transdata";
        op::TransData transdata_op(transdata_name.c_str());
        // set attr
        transdata_op.SetAttr("src_format", "ND");
        transdata_op.SetAttr("dst_format", "FRACTAL_NZ");
        // set input desc
        transdata_op.UpdateInputDesc("src", weight_desc);
        // set output desc
        weight_desc.SetFormat(FORMAT_FRACTAL_NZ);
        Shape shape_nz({(n_dim+15)/16, (k_dim+15)/16, 16, 16});
        weight_desc.SetShape(shape_nz);
        transdata_op.UpdateOutputDesc("dst", weight_desc);
        // update the weight des of mm
        node.UpdateInputDesc(1, weight_desc);
        auto transdata_node = graph->AddNodeByOp(transdata_op);
        // remove adge
        graph->RemoveEdge(*weight_node, weight_index, node, 1);
        // add adge
        graph->AddDataEdge(*weight_node, weight_index, transdata_node, 0);
        graph->AddDataEdge(transdata_node, 0, node, 1);
        matmul_effect++;
    }
    std::cout << "the matmul_match num is  " << matmul_match << std::endl;
    std::cout << "the matmul_effect num is  " << matmul_effect << std::endl;
    return SUCCESS;
}
}

// static Status NzWeightPass(GraphPtr &graph, CustomPassContext &ctx) {
//     auto nodes = graph->GetAllNodes();
//     int inserted = 0;
//     for (auto &node : nodes) {
//         AscendString type;
//         node.GetType(type);
//         std::string op_type(type.GetString());
//         if (op_type != "MatMul" && op_type != "MatMulV2") {
//             continue;
//         }

//         for (int32_t i = 0; i < 2; i++) {
//             auto input_pair = node.GetInDataNodesAndPortIndexs(i);
//             if (!input_pair.first) continue;

//             AscendString in_type;
//             input_pair.first->GetType(in_type);
//             if (std::string(in_type.GetString()) != "Const") continue;

//             TensorDesc desc;
//             if (input_pair.first->GetOutputDesc(0, desc) != 0) continue;
//             auto shape = desc.GetShape();
//             if (shape.GetDimNum() != 2) continue;

//             int64_t k = shape.GetDim(0);
//             int64_t n = shape.GetDim(1);
//             if (k <= 65536 && n <= 65536) continue;
//             if (desc.GetFormat() == FORMAT_FRACTAL_NZ) continue;

//             // Operator trans_op("TransData");
//             // trans_op.SetAttr("src_format", "ND");
//             // trans_op.SetAttr("dst_format", "FRACTAL_NZ");
//             // trans_op.SetAttr("src_subformat", (int64_t)0);
//             // trans_op.SetAttr("dst_subformat", (int64_t)0);
//             // trans_op.SetAttr("groups", (int64_t)1);
//             op::TransData trans_op;
//             trans_op.set_attr_src_format("ND");
//             trans_op.set_attr_dst_format("FRACTAL_NZ");


//             TensorDesc trans_in_desc;
//             trans_in_desc.SetDataType(desc.GetDataType());
//             trans_in_desc.SetFormat(FORMAT_ND);
//             trans_in_desc.SetShape(shape);
//             trans_op.UpdateInputDesc("src", trans_in_desc);

//             TensorDesc trans_out_desc;
//             trans_out_desc.SetDataType(desc.GetDataType());
//             trans_out_desc.SetFormat(FORMAT_FRACTAL_NZ);
//             Shape nz_shape({
//                 (n + 15) / 16,
//                 (k + 15) / 16,
//                 16,
//                 16,
//             });
//             trans_out_desc.SetShape(nz_shape);
//             trans_op.UpdateOutputDesc("dst", trans_out_desc);

//             GNode trans_node = graph->AddNodeByOp(trans_op);

//             graph->RemoveEdge(*input_pair.first, input_pair.second,
//                               node, i);
//             graph->AddDataEdge(*input_pair.first, input_pair.second,
//                                trans_node, 0);
//             graph->AddDataEdge(trans_node, 0, node, i);

//             AscendString const_name;
//             input_pair.first->GetName(const_name);
//             std::cout << "[NzWeightPass] Inserted TransData(ND->FRACTAL_NZ) for "
//                       << "Const[" << const_name.GetString()
//                       << " K=" << k << ",N=" << n << "] -> "
//                       << op_type << std::endl;
//             inserted++;
//         }
//     }
//     std::cout << "[NzWeightPass] Pass done: inserted=" << inserted << std::endl;
//     return SUCCESS;
// }

// REGISTER_CUSTOM_PASS("NzWeightPass")
//     .CustomPassFn(NzWeightPass)
//     .Stage(CustomPassStage::kAfterOriginGraphOptimize);

REGISTER_CUSTOM_PASS("MatMulWeightNZPass")
    .CustomPassFn(MatMulWeightNZPass)
    .Stage(CustomPassStage::kAfterBuiltinFusionPass);
