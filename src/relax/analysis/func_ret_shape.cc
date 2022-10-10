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

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

namespace tvm {
namespace relax {

Expr DeriveFuncRetShape(Array<Var> args, Expr body) {
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> arg_shape_var_set;
  for (auto v : args) {
    if (const ExprNode* s = v->shape_.as<ExprNode>()) {
      Expr shape = GetRef<Expr>(s);
      Array<tir::Var> arg_shape_vars = ShapeVars(shape);
      for (auto v : arg_shape_vars) {
        arg_shape_var_set.insert(v);
      }
    }
  }

  if (const ExprNode* s = body->shape_.as<ExprNode>()) {
    Expr body_shape = GetRef<Expr>(s);
    Array<tir::Var> body_shape_vars = ShapeVars(body_shape);
    for (auto v : body_shape_vars) {
      // if the body shape contains a free var, then we can't
      // be more specific than RuntimeDepShape
      if (arg_shape_var_set.count(v) == 0) {
        return RuntimeDepShape();
      }
    }
    // all vars are defined in the args, so we can use the body shape
    return body_shape;
  }
  return RuntimeDepShape();
}

TVM_REGISTER_GLOBAL(("relax.analysis.derive_func_ret_shape")).set_body_typed(DeriveFuncRetShape);

}  // namespace relax
}  // namespace tvm
