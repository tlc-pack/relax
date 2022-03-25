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
 * \file src/relax/transform/emit_te_rewrite.cc
 * \brief Rewrite Relay Op calls to TIR call
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/te/operation.h>
#include <tvm/tir/function.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/container/optional.h>

#include <string>

namespace tvm {
namespace relax {

// ==================
// EmitTEMutator:
// Rewrite Relay Op to TIR call.

class EmitTEMutator : public ExprMutator {
  public:
    explicit EmitTEMutator(IRModule mod) { mod_ = mod; }

    IRModule Lower() {
      for (auto& p : mod_->functions) {
        Expr func = p.second;
        if (func->IsInstance<FunctionNode>()) {
          func = this->VisitExpr(func);
        }
        builder_->AddFuncToContext(Downcast<BaseFunc>(func), p.first->name_hint);
      }
      return builder_->GetContextIRModule();
    }

    Expr VisitExpr_(const CallNode* call) override {
      Expr expr = VisitExprPostOrder_(call);
      call = expr.as<CallNode>();

      const OpNode* op = call->op.as<OpNode>();
      // TODO: find better way to check if either relay op and relax op
      if (op != nullptr && relay_op_map.count(GetRef<Op>(op)) != 0) {
        Op op_ref = GetRef<Op>(op);

        Array<te::Tensor> te_args;
        const auto* f_te_tensor = runtime::Registry::Get("relax.TETensor");
        for (const Expr& e: call->args) {
          te::Tensor t = (*f_te_tensor)(e, "rxplaceholder");
          te_args.push_back(t);
        }

        Expr output_shape = call->shape();
        Type out_type = call->checked_type();

        Array<te::Tensor> te_out = CallToTE(op_ref, call->attrs, te_args, output_shape, out_type);
        te_args.insert(te_args.end(), te_out.begin(), te_out.end());

        const auto* f_create_func = runtime::Registry::Get("te.CreatePrimFunc");
        te::PrimFunc f = (*f_create_func)(te_args, nullptr);

        GlobalVar gv = builder_->AddFuncToContext(f, op->name);

        const auto* f_make_call_tir = runtime::Registry::Get("relax.op.call_tir");
        Expr ret = (*f_make_call_tir)(gv, Tuple(call->args), output_shape, out_type, nullptr);
        return std::move(ret);
      }

      return GetRef<Expr>(call);
    }
  private:
    IRModule mod_;
    const OpAttrMap<relay::FTVMCompute> relay_op_map = Op::GetAttrMap<relay::FTVMCompute>("FTVMCompute");

    Array<te::Tensor> CallToTE(const Op& op, const Attrs& attrs, const Array<te::Tensor> te_args, const Expr& shape, const Type& out_type) {
      if (op == Op::Get("collapse_sum_like")) {
        // TODO: needs special case because CollapseSumLikeCompute expects out_type to be a TensorTypeNode
        const ShapeExprNode* shape_expr = shape.as<ShapeExprNode>();
        ICHECK(shape_expr != nullptr);
        const DynTensorTypeNode* dyn_type = out_type.as<DynTensorTypeNode>();
        return relay_op_map[op](attrs, te_args, TensorType(shape_expr->values, dyn_type->dtype));
      } else {
        return relay_op_map[op](attrs, te_args, out_type);
      }
    }
};

namespace transform {

Pass EmitTERewrite() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return EmitTEMutator(mod).Lower(); };
  return CreateModulePass(pass_func, 0, "EmitTERewrite", {});
}

TVM_REGISTER_GLOBAL("relax.transform.EmitTERewrite").set_body_typed(EmitTERewrite);

}  // namespace transform

}
}