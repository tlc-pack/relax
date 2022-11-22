/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file vtx.cc
 * \brief vtx operators.
 */

#include "../op_common.h"
#include "../make_op.h"

namespace tvm {
namespace relax {

struct VtxMMAttrs : public tvm::AttrsNode<VtxMMAttrs> {
  bool transpose_a;
  bool transpose_b;
  std::string epilogue_pattern;
  TVM_DECLARE_ATTRS(VtxMMAttrs, "relax.attrs.VtxMMAttrs") {
    TVM_ATTR_FIELD(transpose_a)
        .set_default(false);
    TVM_ATTR_FIELD(transpose_b)
        .set_default(false);
    TVM_ATTR_FIELD(epilogue_pattern)
        .set_default("");
  }
}; 

TVM_REGISTER_NODE_TYPE(VtxMMAttrs);

Type InferTypeVtxMM(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "vtxmm op should have 2 arguments");
  }
  Type lhs_type = call->args[0]->checked_type();
  Type rhs_type = call->args[1]->checked_type();
  auto* t0 = lhs_type.as<DynTensorTypeNode>();
  auto* t1 = rhs_type.as<DynTensorTypeNode>();
  int output_ndim = std::max(t0->ndim, t1->ndim);
  DataType output_dtype = t0->dtype;
  return DynTensorType(output_ndim, output_dtype);
}

Optional<Expr> InferShapeVtxMM(const Call& call, DiagnosticContext diag_ctx) {
    if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                        << "vtxmm op should have 2 arguments");
    }
    Expr lhs_shape = call->args[0]->shape();
    Expr rhs_shape = call->args[1]->shape();
    auto* s0 = lhs_shape.as<ShapeExprNode>();
    auto* s1 = rhs_shape.as<ShapeExprNode>();
    if (s0 && s1) {
        Array<tvm::PrimExpr> output_shape;
        // super hack to suit the vortex case
        output_shape.push_back(s0->values[0]);
        output_shape.push_back(s0->values[1]);
        output_shape.push_back(s1->values[1]);
        return ShapeExpr(output_shape);
    }
    return RuntimeDepShape();
}

RELAY_REGISTER_OP("relax.vtx_mm")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The data tensor")
    .add_argument("weight", "Tensor", "The weight tensor")
    .set_attrs_type<VtxMMAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeVtxMM)
    .set_attr<FInferType>("FInferType", InferTypeVtxMM);

Expr MakeVtxMM(Expr data, Expr weight, bool transpose_a, bool transpose_b, String epilogue_pattern) {
    auto attrs = make_object<VtxMMAttrs>();
    attrs->transpose_a = transpose_a;
    attrs->transpose_b = transpose_b;
    attrs->epilogue_pattern = epilogue_pattern;
    static const Op& op = Op::Get("relax.vtx_mm");
    return Call(op, {data, weight}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.vtx_mm").set_body_typed(MakeVtxMM);

}  // namespace relax
}  // namespace tvm
