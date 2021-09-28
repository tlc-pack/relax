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
/*!
 * \file src/relax/transform/fma_rewrite.cc
 * \brief 
 */
#include <tvm/relax/expr_functor.h>

namespace tvm {
namespace relax {

// ==================
// EwiseFMARewriter
// Example:
// x0 = mul(a, b)
// z0 = add(x0, c)
// -->
// z0 = ewise_fma(a, b, c)

// Example 2: 
// Question: do we want to support this?
// x0 = mul(a, add(k, b))
// z0 = add(x0, c)
// -->
// lv0 = add(k, b)
// z0 = ewise_fma(a, lv0, c)

class EwiseFMARewriter : public DataflowMutator {
	Var VisitVarBinding(const VarBinding& binding, IRBuilder& ir_builder) override {
    static const Op& add_op = Op::Get("relax.add");
    static const Op& multiply_op = Op::Get("relax.multiply");
    static const Op& ewise_fma_op = Op::Get("relax.ewise_fma");

    // TODO: shape & dtype check
    const CallNode* op1 = binding->value.as<CallNode>();
    if (op1 && (op1->op == add_op)) {
      Expr value = LookupVar(Downcast<Var>(op1->args[0]));
      const CallNode* op2 = value.as<CallNode>();
      if (op2 && op2->op == multiply_op) {
        Call fma_call = Call(ewise_fma_op, {op2->args[0], op2->args[1], op1->args[1]}, {}, {});
        return ir_builder->Emit(binding->var, fma_call);
      }
    }
    return ir_builder->Emit(binding);
  }
};

Expr FMARewrite(const Expr& e) {
  return EwiseFMARewriter().Mutate(e);
}

TVM_REGISTER_GLOBAL("relax.transform.fma_rewrite")
.set_body_typed([](Expr expr) {
  return FMARewrite(expr);
});

}  // namespace relax
}  // namespace tvm
