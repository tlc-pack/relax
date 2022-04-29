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
 * \file unary.h
 * \brief shape and type deduction for unary operators.
 */

#ifndef TVM_RELAX_OP_TENSOR_UNARY_H_
#define TVM_RELAX_OP_TENSOR_UNARY_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <vector>

#include "../op_common.h"

namespace tvm {
namespace relax {

Optional<Expr> InferShapeUnique(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Unique op should have 1 argument");
  }
  auto unique_attrs = call->attrs.as<UniqueAttrs>();
  // Only default values of these attributes are supported right now.
  if (unique_attrs->return_counts || unique_attrs->return_inverse || unique_attrs->dim != -1)
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "support for return_inverse, return_counts, and dim is not implemented");
  return relax::RuntimeDepShape(call->span);
}

Type InferTypeUnique(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Unique op should have 1 argument");
  }
  auto* input_ty = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (!input_ty) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Input should be DynTensor, but got "
                       << call->args[0]->checked_type()->GetTypeKey());
  }

  // TODO(prakalp): Add support for return_inverse, return_counts and dim attributes. Only defaults
  // are supported right now.
  auto unique_attrs = call->attrs.as<UniqueAttrs>();
  if (unique_attrs->return_counts || unique_attrs->return_inverse || unique_attrs->dim != -1)
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "support for return_inverse, return_counts, and dim is not implemented");
  return DynTensorType(/*ndim=*/1, input_ty->dtype);
}

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_UNARY_H_
