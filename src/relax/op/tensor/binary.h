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
 * \file binary.h
 * \brief shape and type deduction for binary broadcast operators.
 */

#ifndef TVM_RELAX_OP_TENSOR_BINARY_H_
#define TVM_RELAX_OP_TENSOR_BINARY_H_

#include <tvm/ir/expr.h>
#include <tvm/relax/expr.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relax/type.h>

#include <algorithm>
#include <vector>

#include "../op_common.h"

namespace tvm {
namespace relax {

Optional<Expr> InferShapeBinaryBroadcast(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Binary broadcast op should have 2 arguments");
  }
  Expr lhs_shape = call->args[0]->shape();
  Expr rhs_shape = call->args[1]->shape();
  auto* s0 = lhs_shape.as<ShapeExprNode>();
  auto* s1 = rhs_shape.as<ShapeExprNode>();
  if (s0 && s1) {
    std::vector<PrimExpr> output_shape;
    size_t ndim0 = s0->values.size();
    size_t ndim1 = s1->values.size();
    size_t i = 1;
    for (; i <= std::min(ndim0, ndim1); ++i) {
      PrimExpr dim0 = s0->values[ndim0 - i];
      PrimExpr dim1 = s1->values[ndim1 - i];
      if (EqualConstInt(dim0, 1)) {
        output_shape.push_back(dim1);
      } else if (EqualConstInt(dim1, 1)) {
        output_shape.push_back(dim0);
      } else if (EqualCheck(dim0, dim1)) {
        output_shape.push_back(dim0);
      } else {
        // defer the computation of output shapes to runtime
        // e.g., broadcast Tensor([m, n]), Tensor([k]) -> defer to runtime
        return Call(ExternFunc(String("vm.binary_broadcast_shape_infer")),
                    {call->args[0], call->args[1]}, {}, {});
      }
    }
    size_t max_ndim = std::max(ndim0, ndim1);
    auto& longer_shape = (ndim0 > ndim1) ? s0 : s1;
    for (; i <= max_ndim; ++i) {
      output_shape.push_back(longer_shape->values[max_ndim - i]);
    }
    return ShapeExpr(Array<PrimExpr>(output_shape.rbegin(), output_shape.rend()));
  } else {
    return NullOpt;
  }
}

Type InferTypeBinaryBroadcast(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Binary broadcast op should have 2 arguments");
  }
  Type lhs_type = call->args[0]->checked_type();
  Type rhs_type = call->args[1]->checked_type();
  auto* t0 = lhs_type.as<DynTensorTypeNode>();
  auto* t1 = rhs_type.as<DynTensorTypeNode>();
  if (!t0 || !t1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Both lhs and rhs should be DynTensor for broadcasting, but got "
                       << lhs_type->GetTypeKey() << " and " << rhs_type->GetTypeKey());
  }

  DataType output_dtype;
  if (t0->IsUnknownDtype() || t1->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (t0->dtype != t1->dtype) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Data types " << t0->dtype << " and " << t1->dtype
                       << " must be equal for broadcasting operators");
  } else {
    output_dtype = t0->dtype;
  }

  int output_rank;
  if (t0->IsUnknownRank() || t1->IsUnknownRank()) {
    output_rank = -1;
  } else {
    output_rank = std::max(t0->rank, t1->rank);
  }
  return DynTensorType(output_rank, output_dtype);
}

Optional<Expr> InferShapeBinaryLike(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Binary like op should have 2 arguments");
  }

  return call->args[1]->shape();
}

Type InferTypeBinaryLike(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Binary like op should have 2 arguments");
  }
  return call->args[1]->checked_type();
}

Optional<Expr> InferShapeBinaryNNDense(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Binary nn.dense op should have 2 arguments");
  }

  const ShapeExprNode* tensor_a = call->args[0]->shape().as<ShapeExprNode>();
  const ShapeExprNode* tensor_b = call->args[1]->shape().as<ShapeExprNode>();

  ICHECK(tensor_a != nullptr);
  ICHECK(tensor_b != nullptr);

  // Default set to dense layout
  bool transpose_a = false;
  bool transpose_b = true;
  const auto& mattrs = call->attrs.as<relay::MatmulAttrs>();
  if (mattrs != nullptr) {
    transpose_a = mattrs->transpose_a;
    transpose_b = mattrs->transpose_b;
  }

  Array<PrimExpr> oshape = tensor_a->values;
  const Array<PrimExpr>& wshape = tensor_b->values;
  oshape.Set((oshape.size() - 2), transpose_a ? oshape[oshape.size() - 1] : oshape[oshape.size() - 2]);
  oshape.Set((oshape.size() - 1), transpose_b ? wshape[0] : wshape[1]);

  return ShapeExpr(oshape);
}

Type InferTypeBinaryNNDense(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Binary nn.dense op should have 2 arguments");
  }

  const relay::DenseAttrs* param = call->attrs.as<relay::DenseAttrs>();
  ICHECK(param != nullptr);
  const DynTensorTypeNode* tensor_a = call->args[0]->checked_type().as<relax::DynTensorTypeNode>();
  ICHECK(tensor_a != nullptr);
  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = tensor_a->dtype;
  }
  return DynTensorType(tensor_a->rank, out_dtype);
}

Optional<Expr> InferShapeBinaryTranspose(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "transpose op should have 1 arguments");
  }
  const ShapeExprNode* tensor_a = call->args[0]->shape().as<ShapeExprNode>();

  ICHECK(tensor_a != nullptr);
  
  const auto* param = call->attrs.as<relay::TransposeAttrs>();
  const int ndim = tensor_a->values.size();
  const Array<Integer>& axes = param->axes;
  ICHECK(!axes.defined()) << "only support transpose with no axes set in Relax shape inference for now";
  std::vector<int> int_axes;
  int_axes.reserve(ndim);
  if (!axes.defined()) {
    for (int i = ndim - 1; i >= 0; --i) {
      int_axes.push_back(i);
    }
  }
  Array<PrimExpr> oshape;
  oshape.reserve(ndim);
  for (int axis : int_axes) {
    oshape.push_back(tensor_a->values[axis]);
  }
  return ShapeExpr(oshape);
}

Type InferTypeBinaryTranspose(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "transpose op should have 1 arguments");
  }
  return call->args[0]->checked_type();
}



}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_BINARY_H_
