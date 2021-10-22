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
 * \file src/relax/transform/call_dps_rewrite.cc
 * \brief
 */
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include "../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relax {

// ==================
// CallDPSMutator
// Example:
// y: Tensor[n, m] = rx.call_dps((n, m), op.identity, (x))
// -->
// lv0 = rx.call("relax.builtin.alloc_tensor", [n, m])
// rx.call_packed(op.identity, x, lv0)

class CallDPSMutator : public ExprMutator {
 public:
  explicit CallDPSMutator(IRModule mod) { mod_ = mod; }

  IRModule Lower() {
    IRModule ret_mod = IRModule();
    for (auto& p : mod_->functions) {
      Expr func = p.second;
      if (p.second->IsInstance<FunctionNode>()) {
        func = this->Mutate(p.second);
      }
      ret_mod->Add(p.first, Downcast<BaseFunc>(func));
    }
    return ret_mod;
  }

  Expr VisitExpr_(const CallNode* call) override {
    // post-order mutation
    Expr expr = ExprMutator::VisitExpr_(call);
    call = expr.as<CallNode>();
    // TODO(@yuchen, @altanh): using mutate cause infinite recursion
    // Expr expr = ExprMutator::Mutate(GetRef<Call>(call));

    static const Op& call_dps_op = Op::Get("relax.call_dps");
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");

    if (call->op == call_dps_op) {
      ShapeExpr output_shape = Downcast<ShapeExpr>(call->args[0]);
      Var tensor = builder_->Emit(Call(alloc_tensor_op, {call->args[0]}), "alloc");
      builder_->Emit(Call(call->args[1], {call->args[2], tensor}), "_");
      return tensor;
    }

    return GetRef<Expr>(call);
  }

 private:
  IRModule mod_;
};

TVM_REGISTER_GLOBAL("relax.transform.call_dps_rewrite").set_body_typed([](IRModule mod) {
  return CallDPSMutator(mod).Lower();
});

}  // namespace relax
}  // namespace tvm
