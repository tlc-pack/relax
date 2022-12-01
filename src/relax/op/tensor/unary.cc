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
 * \file unary.cc
 * \brief unary operators.
 */

#include "unary.h"

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(UniqueAttrs);

RELAY_REGISTER_OP("relax.unique")
    .describe(
        "This operation returns the unique elements and the new index of each item in a given "
        "tensor.")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<UniqueAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeUnique)
    .set_attr<FInferType>("FInferType", InferTypeUnique)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.unique");

Expr MakeUnique(Expr data, bool sorted, bool return_inverse, bool return_counts, int dim) {
  auto attrs = make_object<UniqueAttrs>();
  attrs->sorted = sorted;
  attrs->return_inverse = return_inverse;
  attrs->return_counts = return_counts;
  attrs->dim = dim;
  static const Op& op = Op::Get("relax.unique");
  return Call(op, {data}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.unique").set_body_typed(MakeUnique);

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

Expr InferShapeUnique(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Unique op should have 1 argument");
  }
  auto unique_attrs = call->attrs.as<UniqueAttrs>();
  // Only default values of these attributes are supported right now.
  if (unique_attrs->return_counts || unique_attrs->return_inverse || unique_attrs->dim != -1)
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "support for return_inverse, return_counts, and dim is not implemented");
  return RuntimeDepShape(call->span);
}

}  // namespace relax
}  // namespace tvm
