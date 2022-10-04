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

#ifndef TVM_RELAX_OP_NN_POOLING_H_
#define TVM_RELAX_OP_NN_POOLING_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include "../op_common.h"
namespace tvm {
namespace relax {
/*
Optional<Expr> InferShapeMaxPool2d(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "MaxPool2d op should have 1 argument");
  }
  auto attrs = call->attrs.as<MaxPool2dAttrs>();
  Expr shape = call->args[0]->shape();
  auto* s = shape.as<ShapeExprNode>();
  if (s) {
    Array<PrimExpr> output_shape;
    for (int i = 0; i < static_cast<int>(s->values.size()); i++) {
      if (i == static_cast<int>(s->values.size()) - 2) {
        output_shape.push_back((s->values[i] + 2 * attrs->padding[0] -
                                attrs->dilation[0] * (attrs->kernel_size[0] - 1) - 1) /
                                   attrs->stride[0] +
                               1);
      } else if (i == static_cast<int>(s->values.size()) - 1) {
        output_shape.push_back((s->values[i] + 2 * attrs->padding[1] -
                                attrs->dilation[1] * (attrs->kernel_size[1] - 1) - 1) /
                                   attrs->stride[1] +
                               1);
      } else {
        output_shape.push_back(s->values[i]);
      }
    }
    return ShapeExpr(Array<PrimExpr>{output_shape.begin(), output_shape.end()});
  } else {
    return NullOpt;
  }
}
*/

}  // namespace relax
}  // namespace tvm
#endif