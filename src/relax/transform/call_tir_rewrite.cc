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
// lv0: Tensor[n, m] = rx.call_tir(func, (x), (n, m), dtype="float32")
// -->
// gv0 = rx.call("relax.builtin.alloc_tensor", [n, m], dtype="float32")
// rx.call_packed(func, x, gv0)

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
      Array<Expr> outs;
      if (call->shape_) {
        if (call->shape_.value()->IsInstance<ShapeExprNode>()) {
          // single output case
          ShapeExpr output_shape = Downcast<ShapeExpr>(call->shape_.value());
          auto alloc_tensor_attr = make_object<AllocTensorAttrs>();

          if (call->checked_type_.defined()) {
            auto output_type = Downcast<DynTensorType>(call->checked_type_);
            alloc_tensor_attr->dtype = output_type->dtype;
            alloc_tensor_attr->runtime_device_index = 0;
            outs.push_back(builder_->Emit(
                Call(alloc_tensor_op, {output_shape}, Attrs(alloc_tensor_attr)), "alloc"));
          } else {
            LOG(FATAL) << "ValueError: the checked_type_ of call_tir has not populated.";
          }
        } else {
          // multiple output case
          ICHECK(call->shape_.value()->IsInstance<TupleNode>())
              << "call_tir expects ShapeExpr or Tuple as its shape, but got " << call->shape_;
          ICHECK(call->checked_type_->IsInstance<TupleTypeNode>())
              << "call_tir expects DynTensorType or TupleType as its checked type, but got "
              << call->checked_type_;
          Tuple output_shapes = Downcast<Tuple>(call->shape_);
          TupleType output_types = Downcast<TupleType>(call->checked_type_);
          ICHECK(output_shapes->fields.size() == output_types->fields.size())
              << "The output of call_tir should have the same amount of fields in its shape_ and "
                 "checked_type_";
          for (size_t i = 0; i < output_shapes->fields.size(); ++i) {
            ICHECK(output_shapes->fields[i]->IsInstance<ShapeExprNode>())
                << "call_tir expects Tuple of ShapeExprs, but got " << output_shapes->fields[i]
                << " as an element of tuple";
            ICHECK(output_types->fields[i]->IsInstance<DynTensorTypeNode>())
                << "call_tir expects TupleType of DynTensorType, but got "
                << output_types->fields[i] << " as an element of TupleType";
            auto output_type = Downcast<DynTensorType>(output_types->fields[i]);
            auto alloc_tensor_attr = make_object<AllocTensorAttrs>();
            alloc_tensor_attr->dtype = output_type->dtype;
            alloc_tensor_attr->runtime_device_index = 0;
            outs.push_back(builder_->Emit(
                Call(alloc_tensor_op, {Downcast<ShapeExpr>(output_shapes->fields[i])},
                     Attrs(alloc_tensor_attr)),
                "alloc"));
          }
        }
      } else {
        LOG(FATAL) << "ValueError: the shape of call_tir has not populated.";
      }

      Array<Expr> args;
      if (call->args[1].as<TupleNode>()) {
        args = Downcast<Tuple>(call->args[1])->fields;
        args.insert(args.end(), outs.begin(), outs.end());

        if (call->args.size() == 3) {
          builder_->Emit(Call(call->args[0], args), "_");
        } else {
          // unpack semantics
          args.push_back(call->args[3]);
          builder_->Emit(Call(call_tir_dyn_op, {call->args[0], Tuple(args)}), "_");
        }
      } else {
        args = outs;
        args.insert(args.begin(), call->args[1]);
        builder_->Emit(Call(call->args[0], args), "_");
      }

      if (outs.size() == 1) {
        return outs[0];
      }
      return std::move(Tuple(outs));
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
