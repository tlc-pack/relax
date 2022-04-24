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
 * \file ternary.h
 * \brief shape and type deduction for ternary operators.
 */

#ifndef TVM_RELAX_OP_TENSOR_TERNARY_H_
#define TVM_RELAX_OP_TENSOR_TERNARY_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <vector>

#include "../op_common.h"

namespace tvm {
namespace relax {

Optional<Expr> InferShapeEwiseFMA(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 3) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "EwiseFMA op should have 3 arguments");
  }
  Expr shape0 = call->args[0]->shape();
  Expr shape1 = call->args[1]->shape();
  Expr shape2 = call->args[2]->shape();
  auto* s0 = shape0.as<ShapeExprNode>();
  auto* s1 = shape1.as<ShapeExprNode>();
  auto* s2 = shape2.as<ShapeExprNode>();
  if (s0 && s1 && s2) {
    std::vector<PrimExpr> output_shape;
    size_t ndim0 = s0->values.size();
    size_t ndim1 = s1->values.size();
    size_t ndim2 = s2->values.size();
    if (ndim0 != ndim1 || ndim1 != ndim2) {
      LOG(INFO) << ndim0;
      LOG(INFO) << ndim1;
      LOG(INFO) << ndim2;
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "The 3 arguments of EwiseFMA must have the same number of dimensions");
    }
    for (size_t i = 0; i < ndim0; ++i) {
      PrimExpr dim0 = s0->values[i];
      PrimExpr dim1 = s1->values[i];
      PrimExpr dim2 = s2->values[i];
      if (EqualCheck(dim0, dim1) && EqualCheck(dim1, dim2)) {
        output_shape.push_back(dim0);
      } else {
        diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                           << "The 3 arguments of EwiseFMA must have the same shape");
      }
    }
    return ShapeExpr(Array<PrimExpr>(output_shape.begin(), output_shape.end()));
  } else {
    return NullOpt;
  }
}

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

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_TERNARY_H_
