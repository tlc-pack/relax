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

#ifndef TVM_RELAX_OP_NN_NN_H_
#define TVM_RELAX_OP_NN_NN_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include "../op_common.h"
#include "../tensor/unary.h"
namespace tvm {
namespace relax {

StructInfo InferStructInfoFlatten(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Flatten op should have 1 argument");
  }

  auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  if (!sinfo) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Argument should be Tensor, but got "
                     << call->args[0]->struct_info_->GetTypeKey());
  }
  auto* shape = sinfo->shape.as<ShapeExprNode>();
  int output_ndim = shape->values.size();
  if (shape) {
    PrimExpr output_dim = 1;
    for (int i = 1; i < output_ndim; i++) {
      output_dim *= shape->values[i];
    }
    Expr output_shape = ShapeExpr({shape->values[0], output_dim});
    return TensorStructInfo(output_shape, sinfo->dtype);
  } else {
    return TensorStructInfo(sinfo->dtype, output_ndim);
  }
}

StructInfo InferStructInfoDense(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Dense op should have 1 argument");
  }
  
  auto* sinfo0 = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  auto* sinfo1 = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  if (!sinfo0 || !sinfo1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Both arguments should be Tensor, but got "
                     << call->args[0]->struct_info_->GetTypeKey() << " and "
                     << call->args[1]->struct_info_->GetTypeKey());
  }

  // Type deduction
  // data type
  DataType output_dtype;
  if (sinfo0->IsUnknownDtype() || sinfo1->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (sinfo0->dtype != sinfo1->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call->span) << "Data types " << sinfo0->dtype << ", and"
                                                     << sinfo1->dtype << " must be equal for Dense");
  } else {
    output_dtype = sinfo0->dtype;
  }

  // ndims
  int output_ndim;
  if (sinfo0->IsUnknownNdim() || sinfo1->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = sinfo0->ndim;
  }

  // shape
  auto* shape0 = sinfo0->shape.as<ShapeExprNode>();
  auto* shape1 = sinfo1->shape.as<ShapeExprNode>();
  if (shape0 && shape1) {
    size_t ndim0 = shape0->values.size();
    size_t ndim1 = shape1->values.size();
    if (ndim0 != 2 || ndim1 != 2) {
      LOG(INFO) << ndim0;
      LOG(INFO) << ndim1;
      ctx->ReportFatal(Diagnostic::Error(call->span)
                         << "The 2 arguments of Dense must be 2D Tensors");
    }
    if (!EqualCheck(shape0->values[1], shape1->values[1])) {
      ctx->ReportFatal(Diagnostic::Error(call->span)
                         << "The 2 arguments of Dense must have the same number of columns");
    }
    Expr output_shape = ShapeExpr({shape0->values[0], shape1->values[0]});
    return TensorStructInfo(output_shape, output_dtype);
  } else{
    return TensorStructInfo(output_dtype, output_ndim);
  }
}
}  // namespace relax
}  // namespace tvm
#endif