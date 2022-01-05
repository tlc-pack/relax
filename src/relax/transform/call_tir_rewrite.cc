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
 * \file src/relax/transform/call_tir_rewrite.cc
 * \brief Perform explicit tensor allocation for call_tir.
 */
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include "../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relax {

// ==================
// CallTIRMutator
// Perform explicit tensor allocation for call_tir.
// Example:
// lv0: Tensor[n, m] = rx.call_tir((n, m), op.identity, (x))
// -->
// gv0 = rx.call("relax.builtin.alloc_tensor", [n, m])
// rx.call_packed(op.identity, x, gv0)

class CallTIRMutator : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* call) override {
    // post-order mutation
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallNode>();

    static const Op& call_tir_op = Op::Get("relax.call_tir");
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op& call_tir_dyn_op = Op::Get("relax.vm.call_tir_dyn");

    if (call->op == call_tir_op) {
      ShapeExpr output_shape = Downcast<ShapeExpr>(call->args[0]);
      Var tensor = builder_->Emit(Call(alloc_tensor_op, {output_shape}), "alloc");
      Array<Expr> args;
      if (call->args[2].as<TupleNode>()) {
        args = Downcast<Tuple>(call->args[2])->fields;
        args.push_back(tensor);

        if (call->args.size() == 3) {
          builder_->Emit(Call(call->args[1], args), "_");
        } else {
          // unpack semantics
          args.push_back(call->args[3]);
          builder_->Emit(Call(call_tir_dyn_op, {call->args[1], Tuple(args)}), "_");
        }
      } else {
        builder_->Emit(Call(call->args[1], {call->args[2], tensor}), "_");
      }
      return tensor;
    }

    return GetRef<Expr>(call);
  }
};

Expr CallTIRRewrite(const Expr& e) { return CallTIRMutator().VisitExpr(e); }

namespace transform {

Pass CallTIRRewrite() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(CallTIRRewrite(f)); };
  return CreateFunctionPass(pass_func, 0, "CallTIRRewrite", {});
}

TVM_REGISTER_GLOBAL("relax.transform.CallTIRRewrite").set_body_typed(CallTIRRewrite);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
