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
 * \file src/relax/transform/relay_op_rewrite.cc
 * \brief Rewrite Relay Op calls to TIR call
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/target/target.h>
#include <tvm/te/operation.h>
#include <tvm/tir/function.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/container/optional.h>

#include <string>

namespace tvm {
namespace relax {

// ==================
// RelayOpMutator:
// Rewrite Relay Op to TIR call.

class RelayOpMutator : public ExprMutator {
  public:
    explicit RelayOpMutator(IRModule mod, const Target& target) { 
      mod_ = mod;
      target_ = target;
    }

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
      } else if (op != nullptr && relay_strategy_map.count(GetRef<Op>(op)) != 0) {
        Op op = Downcast<Op>(call->op);

        Array<te::Tensor> te_args;
        const auto* f_te_tensor = runtime::Registry::Get("relax.TETensor");
        for (const Expr& e: call->args) {
          te::Tensor t = (*f_te_tensor)(e, "rxplaceholder");
          te_args.push_back(t);
        }

        Expr output_shape = call->shape();
        Type out_type = call->checked_type();

        relay::OpStrategy strategy = relay_strategy_map[op](call->attrs, te_args, out_type, target_);
        auto impl = strategy->specializations[0]->implementations[0];
        auto outs = impl.Compute(call->attrs, te_args, out_type);

        te_args.insert(te_args.end(), outs.begin(), outs.end());
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
    Target target_;
    const OpAttrMap<relay::FTVMCompute> relay_op_map = Op::GetAttrMap<relay::FTVMCompute>("FTVMCompute");
    const OpAttrMap<relay::FTVMStrategy> relay_strategy_map = Op::GetAttrMap<relay::FTVMStrategy>("FTVMStrategy");

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

Pass RelayOpRewrite(const Target& target) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return RelayOpMutator(mod, target).Lower(); };
  return CreateModulePass(pass_func, 0, "RelayOpRewrite", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RelayOpRewrite").set_body_typed([](const Target& target) {
  return RelayOpRewrite(target);
});

}  // namespace transform

}
}