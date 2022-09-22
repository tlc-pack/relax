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
 * \file src/relax/backend/vm/vm_memory_lower.cc
 * \brief Perform memory lowering. Lowers the relax.builtin.alloc_tensor intrinsic to the memory
 * planning intrinsics.
 */
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include <queue>

#include "../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relax {

// ==================
// NaivePlanMemMutator
// Transform module with a naive memory planning:
// 1. Lower the relax.builtin.alloc_tensor op to the memory planning builtin functions, i.e.,
// alloc_storage and alloc_tensor.
// 2. Insert kill_storage and kill_tensor at the end of Function.
// Example:
// x = relax.builtin.alloc_tensor((m, n), relax.attrs.AllocTensorAttrs)
// -->
// gv0 = relax.call_packed("relax.memory.alloc_storage", (m * n), relax.attrs.MemAllocStorageAttrs)
// gv1 = relax.call_packed("relax.memory.alloc_tensor", gv0, (m, n),
// relax.attrs.MemAllocTensorAttrs)
// ...
// relax.call_packed("relax.memory.kill_tensor", gv1)
// relax.call_packed("relax.memory.kill_storage", gv0)

class NaivePlanMemMutator : public ExprMutator {
  Expr ComputeStorageSize(const Expr& shape, const DataType& dtype) const {
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

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call) override {
    // post-order mutation
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallNode>();

    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op& memory_alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const Op& memory_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");

    // TODO(@Lesheng, Yuchen): memory planning
    if (call->op == alloc_tensor_op) {
      ShapeExpr output_shape = Downcast<ShapeExpr>(call->args[0]);
      auto alloc_attrs = call->attrs.as<AllocTensorAttrs>();
      ICHECK(alloc_attrs != nullptr) << "must be AllocTensorAttrs";
      DataType dtype = alloc_attrs->dtype;
      Expr storage_size = ComputeStorageSize(output_shape, dtype);
      auto storage_attr = make_object<MemAllocStorageAttrs>();
      storage_attr->virtual_device_index = alloc_attrs->runtime_device_index;
      storage_attr->storage_scope = "global";
      storage_attr->dtype = dtype;

      Var storage = builder_->Emit(
          Call(memory_alloc_storage_op, {storage_size}, Attrs(storage_attr)), "storage");
      auto tensor_attr = make_object<MemAllocTensorAttrs>();
      tensor_attr->offset = 0;
      tensor_attr->dtype = dtype;
      Expr shape = call->args[0];
      Var tensor = builder_->Emit(
          Call(memory_alloc_tensor_op, {storage, shape}, Attrs(tensor_attr)), "tensor");
      tensors_.insert(tensor);
      tensor2storage_[tensor] = storage;
      return std::move(tensor);
    }

    return GetRef<Expr>(call);
  }

  void CollectLiveTensor(const Expr& expr) {
    if (const auto* node = expr.as<VarNode>()) {
      Var var = Downcast<Var>(expr);
      if (tensors_.count(var) > 0) return_tensors_.insert(Downcast<Var>(var));
      Optional<Expr> val = LookupBinding(var);
      if (val.as<ExprNode>()) {
        CollectLiveTensor(Downcast<Expr>(val));
      }
    }

    if (const auto* node = expr.as<TupleNode>()) {
      for (Expr field : node->fields) {
        CollectLiveTensor(field);
      }
    }
  }

  Expr VisitWithNewScope(const Expr& expr) override {
    static const Op& memory_kill_storage_op = Op::Get("relax.memory.kill_storage");
    static const Op& memory_kill_tensor_op = Op::Get("relax.memory.kill_tensor");

    builder_->BeginBindingBlock();
    Expr ret = this->VisitExpr(expr);

    if (const auto* node = ret.as<SeqExprNode>()) {
      CollectLiveTensor(node->body);
    } else if (const auto* node = ret.as<CallNode>()) {
      for (Expr arg : node->args) {
        if (arg.as<VarNode>()) {
          CollectLiveTensor(arg);
        }
      }
    } else {
      CollectLiveTensor(ret);
    }

    BindingBlock prologue = builder_->EndBlock();
    builder_->BeginBindingBlock();
    for (Var tensor : tensors_) {
      // if the tensor is not used in return, kill that tensor and storage.
      if (return_tensors_.count(tensor) == 0) {
        Var storage = tensor2storage_[tensor];
        builder_->Emit(Call(memory_kill_tensor_op, {tensor}, {}), "kill_tensor");
        builder_->Emit(Call(memory_kill_storage_op, {storage}, {}), "kill_storage");
      }
    }
    BindingBlock memory_kill_block = builder_->EndBlock();

    if (!prologue->bindings.empty()) {
      ret = SeqExpr({prologue}, ret);
    }
    if (const auto* node = ret.as<SeqExprNode>()) {
      Array<BindingBlock> blocks(node->blocks);
      blocks.push_back(memory_kill_block);
      ret = SeqExpr(blocks, node->body);
    }
    tensors_.clear();
    tensor2storage_.clear();
    return_tensors_.clear();
    return ret;
  }

 private:
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> tensors_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> tensor2storage_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> return_tensors_;
};

Expr NaivePlanMem(const Expr& e) { return NaivePlanMemMutator().VisitExpr(e); }

namespace transform {

Pass NaivePlanMemory() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(NaivePlanMem(f)); };
  return CreateFunctionPass(pass_func, 0, "NaivePlanMemory", {});
}

TVM_REGISTER_GLOBAL("relax.transform.NaivePlanMemory").set_body_typed(NaivePlanMemory);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
