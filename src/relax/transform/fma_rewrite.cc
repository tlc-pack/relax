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
 * \brief Perform fused multiply-add rewriting.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

/*! \brief Rewrites the relax.add call to a relax.ewise_fma call when detecting the multiply-add
 * pattern.
 *
 * Example:
 * x0 = mul(a, b)
 * z0 = add(x0, c)
 * -->
 * z0 = ewise_fma(a, b, c)
 *
 * Example 2:
 * Question: do we want to support this?
 * x0 = mul(a, add(k, b))
 * z0 = add(x0, c)
 * -->
 * lv0 = add(k, b)
 * z0 = ewise_fma(a, lv0, c)
 */
class EwiseFMARewriter : public ExprMutator {
  using ExprMutator::VisitExpr_;
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

BindingBlock RewriteFMA(const BindingBlock& block) {
  return EwiseFMARewriter().VisitBindingBlock(block);
}

/*! \brief Performs multiply add fusion. The difference of EwiseFMARewriter and this
 * EwiseFuseFMAMutator class is that this mutator generates a sub function(subgraph) whose body is a
 * CallNode that calls to the relax.ewise_fma op, and rewrites the relax.add call in the main
 * function to calling to the subgraph.
 *
 * Example:
 * Before-transformation IRModule:
 * def main():
 *   x0 = mul(a, b)
 *   z0 = add(x0, c)
 * -->
 * After-transformation IRModule:
 * def ewise_fused(x, y, z):
 *  return relax.ewise_fma(x, y, z)
 *
 * def main():
 *  z0 = ewise_fused(a, b, c)
 */
class EwiseFuseFMAMutator : public ExprMutator {
 public:
  explicit EwiseFuseFMAMutator(IRModule mod) { mod_ = mod; }

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

  using ExprMutator::VisitExpr_;

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
        Var x = Var("x", GetStructInfo(mul->args[0]));
        Var y = Var("y", GetStructInfo(mul->args[1]));
        Var z = Var("z", GetStructInfo(call->args[1]));
        Expr body = Call(ewise_fma_op, {x, y, z});

        String func_name = "ewise_fma_fused";
        Function func = Function({x, y, z}, body, GetStructInfo(call->args[1]));
        Expr normalized = builder_->Normalize(func);
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

Pass RewriteFMA() {
  runtime::TypedPackedFunc<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func =
      [=](DataflowBlock block, IRModule m, PassContext pc) {
        return Downcast<DataflowBlock>(RewriteFMA(block));
      };
  return CreateDataflowBlockPass(pass_func, 2, "RewriteFMA", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RewriteFMA").set_body_typed(RewriteFMA);

Pass FuseFMA() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return EwiseFuseFMAMutator(mod).Transform(); };
  return CreateModulePass(pass_func, 2, "FuseFMA", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FuseFMA").set_body_typed(FuseFMA);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
