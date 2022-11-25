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

#include "../make_op.h"
#include "../op_common.h"

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(VtxMMAttrs);

Type InferTypeVtxMM(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2 && call->args.size() != 3) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "vtxmm op should have 2 or 3 arguments");
  }
  Type lhs_type = call->args[0]->checked_type();
  Type rhs_type = call->args[1]->checked_type();
  auto* t0 = lhs_type.as<DynTensorTypeNode>();
  auto* t1 = rhs_type.as<DynTensorTypeNode>();
  ICHECK_EQ(t0->ndim, t1->ndim);
  int output_ndim = std::max(t0->ndim, t1->ndim);
  DataType output_dtype = t0->dtype;
  return DynTensorType(output_ndim, output_dtype);
}

Optional<Expr> InferShapeVtxMM(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2 && call->args.size() != 3) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "vtxmm op should have 2 or 3 arguments");
  }
  Expr lhs_shape = call->args[0]->shape();
  Expr rhs_shape = call->args[1]->shape();
  auto* s0 = lhs_shape.as<ShapeExprNode>();
  auto* s1 = rhs_shape.as<ShapeExprNode>();
  if (s0 && s1) {
    arith::Analyzer analyzer;
    ICHECK_EQ(s0->values.size(), 3);
    ICHECK_EQ(s1->values.size(), 3);
    // Here we always assume [1, m, k] * [1, n, k]
    ICHECK(analyzer.CanProve(s0->values[0] == 1));
    ICHECK(analyzer.CanProve(s1->values[0] == 1));
    ICHECK(analyzer.CanProve(s0->values[2] == s1->values[2]));
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
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The data tensor")
    .add_argument("weight", "Tensor", "The weight tensor")
    .add_argument("bias", "Tensor", "Optional bias tensor")
    .set_attrs_type<VtxMMAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeVtxMM)
    .set_attr<FInferType>("FInferType", InferTypeVtxMM);

Expr MakeVtxMM(Expr data, Expr weight, Optional<Expr> bias, bool transpose_a, bool transpose_b,
               String epilogue_pattern) {
  static const Op& op = Op::Get("relax.vtx_mm");
  auto attrs = make_object<VtxMMAttrs>();
  attrs->transpose_a = transpose_a;
  attrs->transpose_b = transpose_b;
  attrs->epilogue_pattern = epilogue_pattern;

  Call call;
  if (!bias) {
    call = Call(op, {data, weight}, Attrs(attrs));
  } else {
    call = Call(op, {data, weight, bias.value()}, Attrs(attrs));
  }
  return call;
}

TVM_REGISTER_GLOBAL("relax.op.vtx_mm").set_body_typed(MakeVtxMM);

}  // namespace relax
}  // namespace tvm
