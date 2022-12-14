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
 * \file ternary.cc
 * \brief ternary operators.
 */

#include "ternary.h"

namespace tvm {
namespace relax {


Type InferTypeEwiseFMA(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 3) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "EwiseFMA op should have 3 arguments");
  }
  Type type0 = call->args[0]->checked_type();
  Type type1 = call->args[1]->checked_type();
  Type type2 = call->args[2]->checked_type();
  auto* t0 = type0.as<DynTensorTypeNode>();
  auto* t1 = type1.as<DynTensorTypeNode>();
  auto* t2 = type2.as<DynTensorTypeNode>();
  if (!t0 || !t1 || !t2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The 3 arguments of EwiseFMA should be DynTensor");
  }

  DataType output_dtype;
  if (t0->IsUnknownDtype() || t1->IsUnknownDtype() || t2->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (t0->dtype != t1->dtype || t1->dtype != t2->dtype) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Data types " << t0->dtype << ", " << t1->dtype << ", and " << t2->dtype
                       << " must be equal for EwiseFMA");
  } else {
    output_dtype = t0->dtype;
  }

  int output_ndim;
  if (t0->IsUnknownNdim() || t1->IsUnknownNdim() || t2->IsUnknownNdim()) {
    output_ndim = -1;
  } else {
    output_ndim = t0->ndim;
  }
  return DynTensorType(output_ndim, output_dtype);
}

StructInfo InferStructInfoEwiseFMA(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 3) {
    ctx->ReportFatal(Diagnostic::Error(call->span) << "EwiseFMA op should have 3 arguments");
  }

  auto* t0 = call->args[0]->struct_info_.as<TensorStructInfoNode>();
  auto* t1 = call->args[1]->struct_info_.as<TensorStructInfoNode>();
  auto* t2 = call->args[2]->struct_info_.as<TensorStructInfoNode>();

  if (!t0 || !t1 || !t2) {
    ctx->ReportFatal(Diagnostic::Error(call->span) << "EwiseFMA expects three tensor inputs");
  }

  DataType output_dtype;
  if (t0->IsUnknownDtype() || t1->IsUnknownDtype() || t2->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (t0->dtype != t1->dtype || t1->dtype != t2->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call->span)
                     << "Data types " << t0->dtype << ", " << t1->dtype << ", and " << t2->dtype
                     << " must be equal for EwiseFMA");
  } else {
    output_dtype = t0->dtype;
  }

  auto* s0 = t0->shape.as<ShapeExprNode>();
  auto* s1 = t1->shape.as<ShapeExprNode>();
  auto* s2 = t2->shape.as<ShapeExprNode>();
  if (s0 && s1 && s2) {
    Array<PrimExpr> output_shape;
    size_t ndim0 = s0->values.size();
    size_t ndim1 = s1->values.size();
    size_t ndim2 = s2->values.size();
    if (ndim0 != ndim1 || ndim1 != ndim2) {
      ctx->ReportFatal(Diagnostic::Error(call->span)
                       << "The 3 arguments of EwiseFMA must have the same number of dimensions");
    }
    for (size_t i = 0; i < ndim0; ++i) {
      PrimExpr dim0 = s0->values[i];
      PrimExpr dim1 = s1->values[i];
      PrimExpr dim2 = s2->values[i];
      if (EqualCheck(dim0, dim1) && EqualCheck(dim1, dim2)) {
        output_shape.push_back(dim0);
      } else {
        ctx->ReportFatal(Diagnostic::Error(call->span)
                         << "The 3 arguments of EwiseFMA must have the same shape");
      }
    }
    return TensorStructInfo(ShapeExpr(output_shape), output_dtype);
  }

  int output_ndim;
  if (t0->IsUnknownNdim() || t1->IsUnknownNdim() || t2->IsUnknownNdim()) {
    output_ndim = kUnknownDim;
  } else {
    output_ndim = t0->ndim;
  }
  return TensorStructInfo(output_dtype, output_ndim);
}

RELAY_REGISTER_OP("relax.ewise_fma")
    .set_num_inputs(3)
    .add_argument("e1", "Expr", "The input expression")
    .add_argument("e2", "Expr", "The input expression")
    .add_argument("e3", "Expr", "The input expression")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoEwiseFMA);

Expr MakeEwiseFma(Expr expr1, Expr expr2, Expr expr3) {
  static const Op& op = Op::Get("relax.ewise_fma");
  return Call(op, {expr1, expr2, expr3}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.ewise_fma").set_body_typed(MakeEwiseFma);

}  // namespace relax
}  // namespace tvm
