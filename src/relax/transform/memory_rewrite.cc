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

class ExplicitMemMutator : public DataflowMutator {
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

	Var VisitVarBinding(const VarBinding& binding, IRBuilder& ir_builder) override {
    static const Op& call_dps_op = Op::Get("relax.call_dps");
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");

    const CallNode* op = binding->value.as<CallNode>();
		if(op && op->op == call_dps_op) {
      // switch current DataflowBlock to an impure BindingBlock
      ir_builder->is_dataflow_ = false;
      ShapeExpr output_shape = Downcast<ShapeExpr>(op->args[0]);
      Type arg_type = Downcast<Tuple>(op->args[2])->fields[0]->checked_type();
      Expr output_size = ComputeStorageSize(output_shape, arg_type);
      Var tensor = ir_builder->Emit(Call(alloc_tensor_op, {op->args[0]}));
      return ir_builder->Emit(binding->var, Call(op->args[1], {op->args[2], tensor}));
    }
    return ir_builder->Emit(binding);
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
