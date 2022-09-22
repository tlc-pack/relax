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
 * \brief Perform memory lowering. Lowers the memory planning intrinsics to VM intrinsics.
 */
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include "../../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relax {

// ==================
// VMMemLowerMutator
// Lower the memory planning builtin functions to the VM builtin functions, and remove kill memory
// ops.
// Example: gv0 = relax.call_packed("relax.memory.alloc_storage", (m * n),
// relax.attrs.MemAllocStorageAttrs) gv1 = relax.call_packed("relax.memory.alloc_tensor", gv0, (m,
// n), relax.attrs.MemAllocTensorAttrs)
// ...
// relax.call_packed("relax.memory.kill_tensor", gv1)
// relax.call_packed("relax.memory.kill_storage", gv0)
// -->
// gv0 = relax.call_packed("relax.vm.builtin.alloc_storage", (m * n),
// relax.attrs.VMAllocStorageAttrs)
// gv1 = relax.call_packed("relax.vm.builtin.alloc_tensor", gv0, (m, n),
// relax.attrs.VMAllocTensorAttrs)

class VMMemLowerMutator : public ExprMutator {
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call) override {
    // post-order mutation
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallNode>();

    static const Op& memory_alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const Op& memory_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");
    static const Op& vm_alloc_storage_op = Op::Get("relax.vm.builtin.alloc_storage");
    static const Op& vm_alloc_tensor_op = Op::Get("relax.vm.builtin.alloc_tensor");

    if (call->op == memory_alloc_storage_op) {
      Expr storage_size = call->args[0];
      auto mem_storage_attrs = call->attrs.as<MemAllocStorageAttrs>();
      ICHECK(mem_storage_attrs != nullptr) << "must be MemAllocStorageAttrs";
      auto storage_attrs = make_object<VMAllocStorageAttrs>();
      storage_attrs->dtype = mem_storage_attrs->dtype;
      storage_attrs->runtime_device_index = mem_storage_attrs->virtual_device_index;
      return Call(vm_alloc_storage_op, {storage_size}, Attrs(storage_attrs));
    } else if (call->op == memory_alloc_tensor_op) {
      auto mem_tensor_attrs = call->attrs.as<MemAllocTensorAttrs>();
      ICHECK(mem_tensor_attrs != nullptr) << "must be MemAllocTensorAttrs";
      auto tensor_attrs = make_object<VMAllocTensorAttrs>();
      tensor_attrs->offset = 0;
      tensor_attrs->dtype = mem_tensor_attrs->dtype;
      Expr storage = call->args[0];
      Expr shape = call->args[1];
      return Call(vm_alloc_tensor_op, {storage, shape}, Attrs(tensor_attrs));
    }

    return GetRef<Expr>(call);
  }

  void VisitBinding_(const VarBindingNode* binding) {
    static const Op& memory_kill_storage_op = Op::Get("relax.memory.kill_storage");
    static const Op& memory_kill_tensor_op = Op::Get("relax.memory.kill_tensor");
    if (auto* node = binding->value.as<CallNode>()) {
      if (node->op == memory_kill_storage_op || node->op == memory_kill_tensor_op) return;
    }
    ExprMutator::VisitBinding_(binding);
  }
};

Expr VMMemLower(const Expr& e) { return VMMemLowerMutator().VisitExpr(e); }

namespace transform {

Pass VMMemoryLower() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(VMMemLower(f)); };
  return CreateFunctionPass(pass_func, 0, "VMMemoryLower", {});
}

TVM_REGISTER_GLOBAL("relax.transform.VMMemoryLower").set_body_typed(VMMemoryLower);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
