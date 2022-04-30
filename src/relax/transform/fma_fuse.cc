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
 * \file src/relax/transform/fma_fuse.cc
 * \brief Performs multiply add fusion. The difference of this pass and the fma_rewrite pass is this
 * pass generates a sub function(subgraph) whose body is a CallNode that calls into the
 * relax.ewise_fma op, and rewrites the relax.add call in the main function to calling into the
 * subgraph.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

// ==================
// EwiseFMARewriter
// Example:
// Before-transformation IRModule:
// def main():
//   x0 = mul(a, b)
//   z0 = add(x0, c)
// -->
// After-transformation IRModule:
// def ewise_fused(x, y, z):
//   return relax.ewise_fma(x, y, z)
//
// def main():
//   z0 = ewise_fused(a, b, c)

class EwiseFMAFuseMutator : public ExprMutator {
 public:
  explicit EwiseFMAFuseMutator(IRModule mod) { mod_ = mod; }

  IRModule Transform() {
    for (auto& p : mod_->functions) {
      Expr func = p.second;
      if (func->IsInstance<FunctionNode>()) {
        func = this->VisitExpr(func);
      }
      builder_->AddFunction(Downcast<BaseFunc>(func), p.first->name_hint);
    }
    return builder_->GetContextIRModule();
  }

  Expr VisitExpr_(const CallNode* call) override {
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallNode>();

    static const Op& add_op = Op::Get("relax.add");
    static const Op& multiply_op = Op::Get("relax.multiply");
    static const Op& ewise_fma_op = Op::Get("relax.ewise_fma");

    if (call->op == add_op) {
      Optional<Expr> value = LookupBinding(Downcast<Var>(call->args[0]));
      const CallNode* mul = value.as<CallNode>();
      if (mul && mul->op == multiply_op) {
        // construct a subgraph
        Var x = Var("x", Downcast<Expr>(mul->args[0]->shape_), mul->args[0]->checked_type_);
        Var y = Var("y", Downcast<Expr>(mul->args[1]->shape_), mul->args[1]->checked_type_);
        Var z = Var("z", Downcast<Expr>(call->args[1]->shape_), call->args[1]->checked_type_);
        Expr body = Call(ewise_fma_op, {x, y, z});

        // TODO(@yuchen): avoid creating the unnecessary global_var after #136 is merged
        GlobalVar global_var = GlobalVar("ewise_fma_fused");
        Expr ewise_fma_fused = Function(global_var, {x, y, z}, body, call->args[1]->checked_type_);
        Expr normalized = builder_->Normalize(ewise_fma_fused);
        GlobalVar global_var1 =
            builder_->AddFunction(Downcast<BaseFunc>(normalized), "ewise_fma_fused");

        // construct a call to the subgraph
        Call fma_call = Call(global_var1, {mul->args[0], mul->args[1], call->args[1]}, {}, {});
        return std::move(fma_call);
      }
    }

    return GetRef<Call>(call);
  }

 private:
  IRModule mod_;
};

namespace transform {

Pass FMAFuse() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return EwiseFMAFuseMutator(mod).Transform(); };
  return CreateModulePass(pass_func, 2, "FMAFuse", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FMAFuse").set_body_typed(FMAFuse);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
