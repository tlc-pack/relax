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
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include "../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relax {

// ==================
// MemLowerMutator
// Lower the relax.builtin.alloc_tensor op to VM builtin functions.
// Example:
// x = relax.builtin.alloc_tensor((m, n))
// -->
// gv0 = relax.call_packed("vm.builtin.alloc_storage", (m * n), alignment, device_type,
// relax.attrs.AllocStorageAttrs) gv1 = relax.call_packed("vm.builtin.alloc_tensor", gv0, offset,
// (m, n), relax.attrs.AllocTensorAttrs)

class MemLowerMutator : public ExprMutator {
 public:
  explicit MemLowerMutator(IRModule mod) { mod_ = mod; }

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

  Expr VisitExpr_(const CallNode* call) override {
    // post-order mutation
    Expr expr = ExprMutator::VisitExpr_(call);
    call = expr.as<CallNode>();

    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");

    if (call->op == alloc_tensor_op) {
      ShapeExpr tensor_shape = Downcast<ShapeExpr>(call->args[0]);
      // TODO(@yuchen): Get the type of input x, options: add an attr to relax.builtin.alloc_tensor
      Type tensor_type = DynTensorType(tensor_shape->values.size(), DataType::Float(32));
      Expr storage_size = ComputeStorageSize(tensor_shape, tensor_type);
      ShapeExpr alignment = ShapeExpr({IntImm(DataType::Int(64), 64)});
      ShapeExpr device_type = ShapeExpr({IntImm(DataType::Int(64), 1)});
      auto storage_attr = make_object<AllocStorageAttrs>();
      storage_attr->dtype = DataType::Float(32);
      storage_attr->device_type = 1;

      Var storage =
          builder_->Emit(Call(ExternFunc("vm.builtin.alloc_storage"),
                              {storage_size, alignment}, Attrs(storage_attr)),
                         "storage");

      ShapeExpr offset = ShapeExpr({IntImm(DataType::Int(64), 0)});
      auto tensor_attr = make_object<AllocTensorAttrs>();
      tensor_attr->dtype = DataType::Float(32);
      Expr shape = call->args[0];
      return builder_->Emit(
          Call(ExternFunc("vm.builtin.alloc_tensor"), {storage, offset, shape}, Attrs(tensor_attr)),
          "tensor");
    }

    return GetRef<Expr>(call);
  }

 private:
  IRModule mod_;
};

TVM_REGISTER_GLOBAL("relax.transform.memory_lower").set_body_typed([](IRModule mod) {
  return MemLowerMutator(mod).Lower();
});

}  // namespace relax
}  // namespace tvm
