/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_RELAX_OP_NN_CONVOLUTION_H_
#define TVM_RELAX_OP_NN_CONVOLUTION_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <string>
#include <utility>
#include <vector>

#include "../op_common.h"
namespace tvm {
namespace relax {

template <typename T>
inline Expr MakeConv(Expr data, Expr weight, Array<PrimExpr> strides, Array<PrimExpr> padding,
                     Array<PrimExpr> dilation, int groups, PrimExpr channels,
                     Array<PrimExpr> kernel_size, std::string data_layout,
                     std::string kernel_layout, std::string out_layout, DataType out_dtype,
                     std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}

StructInfo InferStructInfoConv2D(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Binary broadcast op should have 2 arguments");
  }
  auto* sinfo0 = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  auto* sinfo1 = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);

  // Type deduction
  // data type
  DataType output_dtype;
  if (sinfo0->IsUnknownDtype() || sinfo1->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (sinfo0->dtype != sinfo1->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Data types " << sinfo0->dtype << " and " << sinfo1->dtype
                     << " must be equal for Conv2D operator");
  } else {
    output_dtype = sinfo0->dtype;
  }

  // ndims
  int output_ndim;
  if (sinfo0->IsUnknownNdim() || sinfo1->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    size_t ndim0 = sinfo0->ndim;
    size_t ndim1 = sinfo1->ndim;
    if (ndim0 != 4 || ndim1 != 4) {
      LOG(INFO) << ndim0;
      LOG(INFO) << ndim1;
      ctx->ReportFatal(Diagnostic::Error(call)
                         << "The 2 arguments of Conv2d must be 4D Tensors");
    }
    output_ndim = 4;
  }

  // shape
  auto* s0 = sinfo0->shape.as<ShapeExprNode>();
  auto* s1 = sinfo1->shape.as<ShapeExprNode>();
  auto* attrs = call->attrs.as<Conv2DAttrs>();
  if (s0 && s1) {
    std::vector<PrimExpr> output_shape;
    // N
    output_shape.push_back(s0->values[0]);
    // C
    output_shape.push_back(s1->values[0]);
    // H
    output_shape.push_back((s0->values[2] + 2 * attrs->padding[0] -
                            attrs->dilation[0] * (attrs->kernel_size[0] - 1) - 1) /
                               attrs->strides[0] +
                           1);
    // W
    output_shape.push_back((s0->values[3] + 2 * attrs->padding[1] -
                            attrs->dilation[1] * (attrs->kernel_size[1] - 1) - 1) /
                               attrs->strides[1] +
                           1);
    Expr output_shape_expr = ShapeExpr(Array<PrimExpr>{output_shape.begin(), output_shape.end()});
    return TensorStructInfo(output_shape_expr, output_dtype);
  } else {
    return TensorStructInfo(output_dtype, output_ndim);
  }
}

Type InferTypeConv2D(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Conv2d op should have 2 arguments");
  }
  Type type0 = call->args[0]->checked_type();
  Type type1 = call->args[1]->checked_type();
  auto* t0 = type0.as<DynTensorTypeNode>();
  auto* t1 = type1.as<DynTensorTypeNode>();
  if (!t0 || !t1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The 2 arguments of Conv2d should be DynTensor");
  }

  ICHECK(t0->ndim == 4 && t1->ndim == 4) << "Both data and kernel tensors should have a rank of 4.";

  DataType output_dtype;
  if (t0->IsUnknownDtype() || t1->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (t0->dtype != t1->dtype) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Data types " << t0->dtype << ", and"
                                                     << t1->dtype << " must be equal for Conv2d");
  } else {
    output_dtype = t0->dtype;
  }

  int output_ndim;
  if (t0->IsUnknownNdim() || t1->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = t0->ndim;
  }
  return DynTensorType(output_ndim, output_dtype);
}

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_NN_CONVOLUTION_H_
