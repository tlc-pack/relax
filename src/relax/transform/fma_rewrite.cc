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
 * \brief Perform fused multiply add rewriting in dataflow blocks.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

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

class EwiseFMARewriter : public ExprMutator {
  Expr VisitExpr_(const CallNode* call) override {
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallNode>();

    static const Op& add_op = Op::Get("relax.add");
    static const Op& multiply_op = Op::Get("relax.multiply");
    static const Op& ewise_fma_op = Op::Get("relax.ewise_fma");

    if (call->op == add_op) {
      // NOTE: assumes df block is completely SSA
      // FIXME(@altanh, @yuchen): this will crash if args[0] isn't a Var
      Optional<Expr> value = LookupBinding(Downcast<Var>(call->args[0]));
      const CallNode* mul = value.as<CallNode>();
      if (mul && mul->op == multiply_op) {
        Call fma_call = Call(ewise_fma_op, {mul->args[0], mul->args[1], call->args[1]}, {}, {});
        return std::move(fma_call);
      }
    }

    return GetRef<Call>(call);
  }
};

Expr FMARewrite(const Expr& e) { return EwiseFMARewriter().VisitExpr(e); }

namespace transform {

Pass FMARewrite() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(FMARewrite(f)); };
  return CreateFunctionPass(pass_func, 2, "FMARewrite", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FMARewrite").set_body_typed(FMARewrite);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
