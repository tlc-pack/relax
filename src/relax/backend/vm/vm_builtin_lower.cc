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
 * \file src/relax/backend/vm/vm_builtin_lower.cc
 * \brief Lowers most builtin functions and packed calls.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/builtin.h>
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/attrs/shape.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relax {

// This pass lowers most ops to VM specific builtins.
// TODO(relax-team): revisit after PrimValue.
class VMBuiltinLowerMutator : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const CallNode* call_node) final {
    // post-order mutation
    Call call = Downcast<Call>(VisitExprPostOrder_(call_node));

    if (call->op == call_tir_dyn_op_) {
      return CallTIRDyn(call);
    } else if (call->op == make_closure_op_) {
      return MakeClosure(call);
    } else if (call->op == invoke_closure_op_) {
      return InvokeClosure(call);
    } else if (call->op == alloc_tensor_op_) {
      return MakeAllocTensor(call);
    } else {
      return call;
    }
  }

  static ObjectPtr<BuiltinFuncAttrs> DefaultBuiltinAttrs() {
    // initialize with default value
    auto n = make_object<BuiltinFuncAttrs>();
    n->InitBySeq();
    return n;
  }

  Expr ComputeStorageSize(const Expr& shape, const DataType& dtype) const {
    // Question: what if the dtype of tensor_type is unknown?
    // Symbolic/static shape case
    if (auto* shape_expr = shape.as<ShapeExprNode>()) {
      int64_t elem_bytes = runtime::GetVectorBytes(dtype);
      PrimExpr ret = IntImm(DataType::Int(64), elem_bytes);
      for (PrimExpr dim : shape_expr->values) {
        ret = ret * dim;
      }
      return ShapeExpr({ret});
    } else {
      auto attrs = DefaultBuiltinAttrs();
      attrs->dtype_arg = dtype;
      return Call(call_builtin_op_, {builtin_compute_alloc_shape_, Tuple({shape})}, Attrs(attrs),
                  {GetStructInfo(shape)});
    }
  }

  Expr MakeAllocTensor(const Call& call) {
    ShapeExpr output_shape = Downcast<ShapeExpr>(call->args[0]);
    auto alloc_attrs = call->attrs.as<AllocTensorAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be AllocTensorAttrs";
    DataType dtype = alloc_attrs->dtype;
    Expr storage_size = ComputeStorageSize(output_shape, dtype);
    auto storage_attr = make_object<VMAllocStorageAttrs>();
    storage_attr->dtype = dtype;
    storage_attr->runtime_device_index = alloc_attrs->runtime_device_index;
    Var storage =
        builder_->Emit(Call(vm_alloc_storage_op_, {storage_size}, Attrs(storage_attr)), "storage");
    auto tensor_attr = make_object<VMAllocTensorAttrs>();
    tensor_attr->offset = 0;
    tensor_attr->dtype = dtype;
    Expr shape = call->args[0];
    return Call(vm_alloc_tensor_op_, {storage, shape}, Attrs(tensor_attr));
  }

  Expr CallTIRDyn(const Call& call_node) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<GlobalVarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());
    Array<Expr> args;

    auto tir_args = Downcast<Tuple>(call_node->args[1]);
    args.push_back(call_node->args[0]);
    for (Expr arg : tir_args->fields) {
      args.push_back(arg);
    }
    auto attrs = DefaultBuiltinAttrs();
    return Call(call_builtin_op_, {builtin_call_tir_dyn_, Tuple(args)}, Attrs(attrs));
  }

  Expr MakeClosure(const Call& call_node) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<GlobalVarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());

    Array<Expr> args;
    auto func = call_node->args[0];
    auto closure_args = Downcast<Tuple>(call_node->args[1]);

    args.push_back(func);
    for (Expr arg : closure_args->fields) {
      args.push_back(arg);
    }
    auto attrs = DefaultBuiltinAttrs();

    return Call(call_builtin_op_, {builtin_make_closure_, Tuple(args)}, Attrs(attrs),
                {object_sinfo_});
  }

  Expr InvokeClosure(const Call& call_node) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<VarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());

    Array<Expr> args;

    args.push_back(call_node->args[0]);

    // args for the invoke_closure
    auto invoke_closure_args = Downcast<Tuple>(call_node->args[1]);
    for (Expr arg : invoke_closure_args->fields) {
      args.push_back(arg);
    }
    auto attrs = DefaultBuiltinAttrs();
    attrs->require_ctx = true;
    return Call(call_builtin_op_, {builtin_invoke_closure_, Tuple(args)}, Attrs(attrs),
                {object_sinfo_});
  }

  const Op& call_builtin_op_ = Op::Get("relax.call_builtin");
  const StructInfo object_sinfo_ = ObjectStructInfo();
  // object to pattern match.
  const Op& call_tir_dyn_op_ = Op::Get("relax.vm.call_tir_dyn");
  const Op& make_closure_op_ = Op::Get("relax.make_closure");
  const Op& invoke_closure_op_ = Op::Get("relax.invoke_closure");
  const Op& alloc_tensor_op_ = Op::Get("relax.builtin.alloc_tensor");
  // functions to lower to
  const Op& vm_alloc_storage_op_ = Op::Get("relax.vm.alloc_storage");
  const Op& vm_alloc_tensor_op_ = Op::Get("relax.vm.alloc_tensor");
  // Function to compute allocated shape.
  const ExternFunc builtin_compute_alloc_shape_{"vm.builtin.compute_alloc_shape"};
  const ExternFunc builtin_call_tir_dyn_{"vm.builtin.call_tir_dyn"};
  const ExternFunc builtin_make_closure_{"vm.builtin.make_closure"};
  const ExternFunc builtin_invoke_closure_{"vm.builtin.invoke_closure"};
};

Expr VMBuiltinLower(const Expr& e) { return VMBuiltinLowerMutator().VisitExpr(e); }

namespace transform {

Pass VMBuiltinLower() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(VMBuiltinLower(f)); };
  return CreateFunctionPass(pass_func, 0, "VMBuiltinLower", {});
}

TVM_REGISTER_GLOBAL("relax.transform.VMBuiltinLower").set_body_typed(VMBuiltinLower);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
