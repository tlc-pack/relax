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
      storages_.push_back(storage);
      alias_map_[storage.get()] = storage.get();

      auto tensor_attr = make_object<MemAllocTensorAttrs>();
      tensor_attr->offset = 0;
      tensor_attr->dtype = dtype;
      Expr shape = call->args[0];
      return Call(memory_alloc_tensor_op, {storage, shape}, Attrs(tensor_attr));
    }

    return GetRef<Expr>(call);
  }

  void VisitBinding_(const VarBindingNode* binding) {
    Expr new_value = this->VisitExpr(binding->value);
    Var new_var = this->VisitVarDef(binding->var);

    static const Op& memory_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");
    static const Op& make_closure_op = Op::Get("relax.make_closure");

    if (const auto* node = new_value.as<CallNode>()) {
      if (node->op == memory_alloc_tensor_op) {
        tensors_.push_back(new_var);
        alias_map_[new_var.get()] = new_var.get();
        this->builder_->Emit(VarBinding(new_var, new_value));
        return;
      } else if (node->op == make_closure_op) {
        alias_map_[new_var.get()] = node;
      }
    } else if (const auto* node = new_value.as<VarNode>()) {
      // handle aliasing
      auto iter = alias_map_.find(node);
      if (iter != alias_map_.end()) {
        alias_map_[new_var.get()] = iter->second;
      }
    } else if (const auto* node = new_value.as<TupleNode>()) {
      alias_map_[new_var.get()] = node;
    }

    this->builder_->Emit(GetRef<VarBinding>(binding));
  }

  void CollectLiveObject(const Expr& expr) {
    if (const auto* node = expr.as<TupleNode>()) {
      for (Expr field : node->fields) {
        CollectLiveObject(field);
      }
    } else if (const auto* node = expr.as<CallNode>()) {
      for (Expr arg : node->args) {
        CollectLiveObject(arg);
      }
    } else if (const auto* node = expr.as<VarNode>()) {
      if (!expr->checked_type_.defined()) return;  // TODO(@Lesheng Jin): Remove it?
      if (expr->checked_type().as<DynTensorTypeNode>()) {
        // It's a tensor(TODO(@Lesheng Jin): function parameter)
        const ExprNode* expr_node = alias_map_[node];
        if (const VarNode* var_node = GetRef<Expr>(expr_node).as<VarNode>())
          if (std::find(tensors_.begin(), tensors_.end(), GetRef<Var>(var_node)) != tensors_.end())
            live_objects_.insert(GetRef<Var>(var_node));
      } else if (expr->checked_type().as<ObjectTypeNode>()) {
        // It can be a storage or Closure
        if (alias_map_.count(node) > 0) {
          const ExprNode* expr_node = alias_map_[node];
          if (const VarNode* var_node = GetRef<Expr>(expr_node).as<VarNode>()) {
            if (std::find(storages_.begin(), storages_.end(), GetRef<Var>(var_node)) !=
                storages_.end())
              // it is a storage object
              live_objects_.insert(GetRef<Var>(var_node));
          } else {
            // it is a closure, visit args of make_closure
            const CallNode* closure_node = static_cast<const CallNode*>(expr_node);
            for (Expr arg : closure_node->args) {
              CollectLiveObject(arg);
            }
          }
        }
      } else if (expr->checked_type().as<TupleTypeNode>()) {
        // It's a Tuple, redirect to the original TupleNode
        auto* tuple_node = alias_map_[node];
        CollectLiveObject(GetRef<Expr>(tuple_node));
      } else {
        // Do nothing to ShapeType and FuncType
      }
    }
  }

  Expr VisitWithNewScope(const Expr& expr) override {
    static const Op& memory_kill_storage_op = Op::Get("relax.memory.kill_storage");
    static const Op& memory_kill_tensor_op = Op::Get("relax.memory.kill_tensor");

    builder_->BeginBindingBlock();
    Expr ret = this->VisitExpr(expr);
    BindingBlock prologue = builder_->EndBlock();

    if (ret.as<TupleNode>()) {
      CollectLiveObject(ret);
    } else if (ret.as<VarNode>()) {
      CollectLiveObject(ret);
    } else if (const auto* node = ret.as<CallNode>()) {
      for (Expr arg : node->args) {
        CollectLiveObject(arg);
      }
    } else if (const auto* node = ret.as<SeqExprNode>()) {
      CollectLiveObject(node->body);
    } else {
      // Constant: Do nothing
      // DataflowVar: Should not appear in the module
      // ShapeExprNode: Do nothing
      // RuntimeDepShapeNode: Should not be the return
      // ExternFunc: Do nothing
      // GlobalVar: Do nothing
      // FunctionNode: Do nothing
      // IfNode: Should not be the return
      // OpNode: Should not be the return
    }

    builder_->BeginBindingBlock();
    for (size_t i = 0; i < tensors_.size(); i++) {
      Var tensor = tensors_[i];
      Var storage = storages_[i];
      if (live_objects_.count(tensor) == 0)
        builder_->Emit(Call(memory_kill_tensor_op, {tensor}, {}), "kill_tensor");
      if (live_objects_.count(tensor) == 0 && live_objects_.count(storage) == 0)
        builder_->Emit(Call(memory_kill_storage_op, {storage}, {}), "kill_storage");
    }
    BindingBlock memory_kill_block = builder_->EndBlock();

    if (auto* node = ret.as<SeqExprNode>()) {
      Array<BindingBlock> blocks = node->blocks;
      blocks.push_back(memory_kill_block);
      ret = SeqExpr(blocks, node->body);
    } else if (!prologue->bindings.empty()) {
      ret = (memory_kill_block->bindings.empty()) ? SeqExpr({prologue}, ret)
                                                  : SeqExpr({prologue, memory_kill_block}, ret);
    } else {
      // If ret is not a SeqExpr node and there are no bindings in prologue,
      // there is nothing to kill
      ICHECK(memory_kill_block->bindings.empty());
    }

    tensors_.clear();
    storages_.clear();
    live_objects_.clear();
    alias_map_.clear();
    return ret;
  }

 private:
  std::vector<Var> tensors_;
  std::vector<Var> storages_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> live_objects_;
  std::unordered_map<const VarNode*, const ExprNode*> alias_map_;
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
