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
 * \file src/relax/transform/memory_rewrite.cc
 * \brief
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include "../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relax {

// ==================
// ExplicitMemMutator
// Example:
// y: Tensor[n, m] = rx.call_dps((n, m), op.identity, (x))
// -->
// lv0 = rx.call("relax.builtin.alloc_tensor", [n, m])
// rx.call_packed(op.identity, x, lv0)

class ExplicitMemMutator : public ExprMutator {
  Expr ComputeStorageSize(const Expr& shape, const Type& type) const {
    DynTensorType tensor_type = Downcast<DynTensorType>(type);
    DataType dtype = DataType(tensor_type->dtype);
    // Question: what if the dtype of tensor_type is unknown?
    // Symbolic/static shape case
    if (auto* shape_expr = shape.as<ShapeExprNode>()) {
      PrimExpr num = PrimExpr(dtype.bits()) * PrimExpr(dtype.lanes());
      PrimExpr add = num + 7;
      PrimExpr ret = 1;
      for (PrimExpr dim : shape_expr->values) {
        ret = ret * dim;
      }
      ret = ret * (add / PrimExpr(8));
      return ShapeExpr({ret});
    }
    // Fully dynamic shape case
    // will need to dedup with ComputeStorageInRelay when we upstream
    Expr prod = relay::Prod(shape, Array<Integer>(nullptr), false, false);
    Expr num = relay::MakeConstantScalar(DataType::Int(64), dtype.bits() * dtype.lanes());
    Expr add = relay::Add(num, relay::MakeConstantScalar(DataType::Int(64), 7));
    Expr div = relay::MakeConstantScalar(DataType::Int(64), 8);
    Expr ret = relay::Multiply(prod, relay::Divide(add, div));
    return ret;
  }

  BindingBlock VisitBindingBlock(const BindingBlock& block) {
    builder_->BeginBindingBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
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
      Type arg_type = Downcast<Tuple>(call->args[2])->fields[0]->checked_type();
      Expr output_size = ComputeStorageSize(output_shape, arg_type);
      Var tensor = builder_->Emit(Call(alloc_tensor_op, {call->args[0]}), "alloc");
      builder_->Emit(Call(call->args[1], {call->args[2], tensor}), "_");
      return tensor;
    }

    return GetRef<Expr>(call);
  }
};

Expr ExplicitMemRewrite(const Expr& e) { 
  return ExplicitMemMutator().Mutate(e); 
}

TVM_REGISTER_GLOBAL("relax.transform.explicit_memory_rewrite")
.set_body_typed([](Expr expr) {
  return ExplicitMemRewrite(expr);
});

}  // namespace relax
}  // namespace tvm
