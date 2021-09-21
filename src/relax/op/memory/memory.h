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
 * \file memory.h
 * \brief shape and type deduction for memory related operators.
 */
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

Optional<Expr> InferShapeAllocStorage(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "AllocStorage op should have 1 argument");
  }
  return call->args[0];
}

Type InferTypeAllocStorage(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "AllocStorage op should have 1 argument");
  }
  DataType output_dtype;
  return DynTensorType(1, output_dtype);
}

Optional<Expr> InferShapeAllocTensor(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 3) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "AllocTensor op should have 3 argument");
  }
  return call->args[2];
}

Type InferTypeAllocTensor(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 3) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "AllocTensor op should have 3 argument");
  }
  int output_rank;
  if (auto* shape = call->args[2].as<ShapeExprNode>()) {
    output_rank = shape->values.size();
  } else {
    output_rank = -1;
  }
  DataType output_dtype;
  return DynTensorType(output_rank, output_dtype);
}

}  // namespace relax
}  // namespace tvm
